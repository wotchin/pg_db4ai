/* SPDX-License-Identifier: MulanPSL-2.0
 * Immutable dense replay store for normalized training rows.
 * PG owns the memory and temporary file lifetime. No SQL, table pages, or
 * visibility rules are bypassed: only the already-authorized query result is
 * stored here. A block is a contiguous logical range, not a heap page.
 */
#include "postgres.h"
#include "db4ai.h"
#include "miscadmin.h"
#include "storage/buffile.h"
#include "utils/memutils.h"

struct Db4aiRowStore
{
    MemoryContext context;
    MemoryContextCallback cleanup;
    BufFile *file;
    double **chunks;
    double *tail;
    double *readbuf;
    int ncolumns;
    int chunk_rows;
    int max_chunks;
    int nchunks;
    int tail_rows;
    int64 nrows;
    int64 read_chunk;
    int read_rows;
    bool sealed;
    Size rowbytes;
    Size chunkbytes;
    uint64 bytes_read;
    uint64 bytes_written;
    uint64 seeks;
};

static void
close_file(void *arg)
{
    Db4aiRowStore *s = arg;
    if (s->file != NULL)
    {
        BufFileClose(s->file);
        s->file = NULL;
    }
}

Db4aiRowStore *
db4ai_rows_create(int ncolumns, Size budget)
{
    MemoryContext context;
    MemoryContext old;
    Db4aiRowStore *s;
    Size rowbytes;
    Size available;

    if (ncolumns < 1 || ncolumns > 16386)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("invalid dense replay row width")));
    rowbytes = (Size) ncolumns * sizeof(double);
    if (budget < rowbytes * 2 + 4096)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("dense replay row exceeds memory budget")));
    context = AllocSetContextCreate(CurrentMemoryContext, "DB4AI dense replay", ALLOCSET_DEFAULT_SIZES);
    old = MemoryContextSwitchTo(context);
    s = palloc0(sizeof(*s));
    s->context = context;
    s->ncolumns = ncolumns;
    s->rowbytes = rowbytes;
    /* Leave room for pointer inventory and allocator metadata. A chunk is
     * deliberately independent of shuffle block and minibatch sizes. */
    available = budget - 4096;
    s->chunk_rows = (int) Max((Size) 1, Min((Size) 4096, Min(available / 4, (Size) 262144) / rowbytes));
    s->chunkbytes = (Size) s->chunk_rows * rowbytes;
    s->max_chunks = (int) Min((Size) INT_MAX, available / (s->chunkbytes + sizeof(double *) + 32));
    s->max_chunks = Max(1, s->max_chunks);
    s->chunks = palloc0((Size) s->max_chunks * sizeof(double *));
    s->read_chunk = -1;
    s->cleanup.func = close_file;
    s->cleanup.arg = s;
    MemoryContextRegisterResetCallback(context, &s->cleanup);
    MemoryContextSwitchTo(old);
    return s;
}

static void
spill(Db4aiRowStore *s)
{
    int i;
    s->file = BufFileCreateTemp(false);
    for (i = 0; i < s->nchunks; i++)
    {
        int count = (int) Min((int64) s->chunk_rows, s->nrows - (int64) i * s->chunk_rows);
        CHECK_FOR_INTERRUPTS();
        BufFileWrite(s->file, s->chunks[i], (Size) count * s->rowbytes);
        s->bytes_written += (Size) count * s->rowbytes;
        pfree(s->chunks[i]);
    }
    pfree(s->chunks);
    s->chunks = NULL;
    s->nchunks = 0;
    s->tail = palloc(s->chunkbytes);
    s->tail_rows = 0;
}

void
db4ai_rows_append(Db4aiRowStore *s, const double *row)
{
    MemoryContext old;
    int offset;
    if (s->sealed)
        elog(ERROR, "cannot append to sealed DB4AI replay store");
    if (s->nrows >= PG_INT64_MAX / (int64) s->rowbytes)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("dense replay file exceeds offset limits")));
    old = MemoryContextSwitchTo(s->context);
    offset = (int) (s->nrows % s->chunk_rows);
    if (s->file == NULL)
    {
        if (offset == 0)
        {
            if (s->nchunks == s->max_chunks)
                spill(s);
            else
                s->chunks[s->nchunks++] = palloc(s->chunkbytes);
        }
        if (s->file == NULL)
            memcpy(s->chunks[s->nchunks - 1] + (Size) offset * s->ncolumns, row, s->rowbytes);
    }
    if (s->file != NULL)
    {
        memcpy(s->tail + (Size) s->tail_rows * s->ncolumns, row, s->rowbytes);
        if (++s->tail_rows == s->chunk_rows)
        {
            BufFileWrite(s->file, s->tail, s->chunkbytes);
            s->bytes_written += s->chunkbytes;
            s->tail_rows = 0;
        }
    }
    s->nrows++;
    MemoryContextSwitchTo(old);
}

void
db4ai_rows_seal(Db4aiRowStore *s)
{
    MemoryContext old;
    if (s->sealed)
        return;
    old = MemoryContextSwitchTo(s->context);
    if (s->file != NULL)
    {
        if (s->tail_rows)
        {
            BufFileWrite(s->file, s->tail, (Size) s->tail_rows * s->rowbytes);
            s->bytes_written += (Size) s->tail_rows * s->rowbytes;
        }
        s->readbuf = s->tail; /* Reuse the buffer after the last append. */
        s->tail = NULL;
    }
    s->sealed = true;
    MemoryContextSwitchTo(old);
}

/* Returned storage is borrowed until the next store read; never persist it. */
int
db4ai_rows_slice(Db4aiRowStore *s, int64 start, int maximum, const double **rows)
{
    int64 chunk;
    int offset;
    int count;
    if (!s->sealed || start < 0 || start > s->nrows || maximum < 0)
        elog(ERROR, "invalid dense replay read");
    if (start == s->nrows || maximum == 0)
    {
        *rows = NULL;
        return 0;
    }
    chunk = start / s->chunk_rows;
    offset = (int) (start % s->chunk_rows);
    count = (int) Min((int64) Min(maximum, s->chunk_rows - offset), s->nrows - start);
    if (s->file == NULL)
        *rows = s->chunks[chunk] + (Size) offset * s->ncolumns;
    else
    {
        if (s->read_chunk != chunk)
        {
            Size length;
            off_t position = (off_t) (chunk * s->chunkbytes);
            s->read_rows = (int) Min((int64) s->chunk_rows, s->nrows - chunk * s->chunk_rows);
            length = (Size) s->read_rows * s->rowbytes;
            CHECK_FOR_INTERRUPTS();
            if (BufFileSeek(s->file, 0, position, SEEK_SET) != 0)
                ereport(ERROR, (errcode_for_file_access(), errmsg("cannot seek DB4AI replay file")));
            if (BufFileRead(s->file, s->readbuf, length) != length)
                ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("short read in DB4AI replay file")));
            s->read_chunk = chunk;
            s->bytes_read += length;
            s->seeks++;
        }
        *rows = s->readbuf + (Size) offset * s->ncolumns;
    }
    return count;
}

int
db4ai_rows_read(Db4aiRowStore *s, int64 start, int maximum, double *out)
{
    int total = 0;
    if (!s->sealed || start < 0 || start > s->nrows || maximum < 0)
        elog(ERROR, "invalid dense replay range");
    /* Block shuffle must read exactly the requested block, not a much larger
     * replay cache chunk for each random block (which amplifies spilled I/O).
     * BufFile still provides its page buffering and segmented-file handling. */
    if (s->file != NULL && start < s->nrows && maximum > 0)
    {
        int count = (int) Min((int64) maximum, s->nrows - start);
        Size length = (Size) count * s->rowbytes;
        CHECK_FOR_INTERRUPTS();
        if (BufFileSeek(s->file, 0, (off_t) (start * s->rowbytes), SEEK_SET) != 0)
            ereport(ERROR, (errcode_for_file_access(), errmsg("cannot seek DB4AI replay range")));
        if (BufFileRead(s->file, out, length) != length)
            ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("short DB4AI replay range")));
        s->bytes_read += length;
        s->seeks++;
        return count;
    }
    while (total < maximum)
    {
        const double *rows;
        int count = db4ai_rows_slice(s, start + total, maximum - total, &rows);
        if (!count)
            break;
        memcpy(out + (Size) total * s->ncolumns, rows, (Size) count * s->rowbytes);
        total += count;
    }
    return total;
}

bool
db4ai_rows_spilled(Db4aiRowStore *s)
{
    return s->file != NULL;
}

void
db4ai_rows_stats(Db4aiRowStore *s, uint64 *read, uint64 *written, uint64 *seeks)
{
    *read = s->bytes_read;
    *written = s->bytes_written;
    *seeks = s->seeks;
}

void
db4ai_rows_free(Db4aiRowStore *s)
{
    MemoryContextDelete(s->context);
}
