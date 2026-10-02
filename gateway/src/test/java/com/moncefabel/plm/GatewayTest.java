package com.moncefabel.plm;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GatewayTest {

    @Test
    void extractsAndTrimsQuestion() {
        assertEquals("pompe hydraulique", AskResource.extractQuestion("{\"question\":\"  pompe hydraulique \"}"));
    }

    @Test
    void rejectsInvalidBodies() {
        assertThrows(IllegalArgumentException.class, () -> AskResource.extractQuestion(""));
        assertThrows(IllegalArgumentException.class, () -> AskResource.extractQuestion("not json"));
        assertThrows(IllegalArgumentException.class, () -> AskResource.extractQuestion("{\"question\":\"x\"}"));
    }

    @Test
    void countsSources() {
        assertEquals(2, AskResource.countSources("{\"sources\":[{},{}]}"));
        assertEquals(0, AskResource.countSources("garbage"));
    }

    @Test
    void historyIsBoundedAndMostRecentFirst() {
        QueryHistory h = new QueryHistory(2);
        h.record(new QueryHistory.Entry("a", 1, 10, Instant.now()));
        h.record(new QueryHistory.Entry("b", 1, 10, Instant.now()));
        h.record(new QueryHistory.Entry("c", 1, 10, Instant.now()));
        List<QueryHistory.Entry> s = h.snapshot();
        assertEquals(2, s.size());
        assertEquals("c", s.get(0).question());
    }
}
