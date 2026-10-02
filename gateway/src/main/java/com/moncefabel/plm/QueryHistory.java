package com.moncefabel.plm;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/** Thread-safe, bounded, in-memory log of the last questions asked. */
public final class QueryHistory {

    /** One recorded question. */
    public record Entry(String question, int resultCount, long latencyMs, Instant at) { }

    private static final QueryHistory INSTANCE = new QueryHistory(20);

    private final int capacity;
    private final Deque<Entry> entries = new ArrayDeque<>();

    QueryHistory(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        this.capacity = capacity;
    }

    public static QueryHistory get() {
        return INSTANCE;
    }

    public synchronized void record(Entry entry) {
        entries.addFirst(entry);
        while (entries.size() > capacity) {
            entries.removeLast();
        }
    }

    /** Most recent first. */
    public synchronized List<Entry> snapshot() {
        return new ArrayList<>(entries);
    }
}
