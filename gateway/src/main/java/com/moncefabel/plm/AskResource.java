package com.moncefabel.plm;

import jakarta.json.Json;
import jakarta.json.JsonException;
import jakarta.json.JsonObject;
import jakarta.json.JsonReader;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.io.StringReader;
import java.time.Instant;

/**
 * POST /api/ask : validates the question, forwards it to the RAG service and
 * records it in the query history.
 */
@Path("/ask")
public class AskResource {

    static final int MIN_LENGTH = 3;
    static final int MAX_LENGTH = 500;

    private final RagClient rag;

    public AskResource() {
        this(RagClient.fromEnvironment());
    }

    AskResource(RagClient rag) {
        this.rag = rag;
    }

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response ask(String body) {
        String question;
        try {
            question = extractQuestion(body);
        } catch (IllegalArgumentException e) {
            return error(Response.Status.BAD_REQUEST, e.getMessage());
        }

        String payload = Json.createObjectBuilder()
                .add("question", question)
                .add("k", 5)
                .add("mode", "hybrid")
                .build()
                .toString();

        long start = System.nanoTime();
        try {
            String result = rag.ask(payload);
            long latencyMs = (System.nanoTime() - start) / 1_000_000;
            QueryHistory.get().record(new QueryHistory.Entry(
                    question, countSources(result), latencyMs, Instant.now()));
            return Response.ok(result, MediaType.APPLICATION_JSON).build();
        } catch (IOException e) {
            return error(Response.Status.BAD_GATEWAY, "RAG service unavailable");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return error(Response.Status.SERVICE_UNAVAILABLE, "Request interrupted");
        }
    }

    /** Parses {"question": "..."} and enforces the length limits. */
    static String extractQuestion(String body) {
        if (body == null || body.isBlank()) {
            throw new IllegalArgumentException("Empty request body");
        }
        try (JsonReader reader = Json.createReader(new StringReader(body))) {
            JsonObject json = reader.readObject();
            String question = json.getString("question", "").strip();
            if (question.length() < MIN_LENGTH || question.length() > MAX_LENGTH) {
                throw new IllegalArgumentException(
                        "question must contain between " + MIN_LENGTH + " and " + MAX_LENGTH + " characters");
            }
            return question;
        } catch (JsonException | ClassCastException e) {
            throw new IllegalArgumentException("Invalid JSON body");
        }
    }

    static int countSources(String ragJson) {
        try (JsonReader reader = Json.createReader(new StringReader(ragJson))) {
            return reader.readObject().getJsonArray("sources").size();
        } catch (RuntimeException e) {
            return 0;
        }
    }

    private static Response error(Response.Status status, String message) {
        String json = Json.createObjectBuilder().add("error", message).build().toString();
        return Response.status(status).entity(json).type(MediaType.APPLICATION_JSON).build();
    }
}
