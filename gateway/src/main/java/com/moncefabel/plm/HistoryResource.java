package com.moncefabel.plm;

import jakarta.json.Json;
import jakarta.json.JsonArrayBuilder;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

/** GET /api/history : the last questions, with result count and latency. */
@Path("/history")
public class HistoryResource {

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public String history() {
        JsonArrayBuilder array = Json.createArrayBuilder();
        for (QueryHistory.Entry e : QueryHistory.get().snapshot()) {
            array.add(Json.createObjectBuilder()
                    .add("question", e.question())
                    .add("resultCount", e.resultCount())
                    .add("latencyMs", e.latencyMs())
                    .add("at", e.at().toString()));
        }
        return array.build().toString();
    }
}
