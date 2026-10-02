package com.moncefabel.plm;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** HTTP client for the Python RAG service (FastAPI). */
public class RagClient {

    private final URI baseUri;
    private final HttpClient http;

    public RagClient(String baseUrl) {
        this.baseUri = URI.create(baseUrl.endsWith("/") ? baseUrl : baseUrl + "/");
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    /** Reads RAG_URL, defaulting to the docker-compose service name. */
    public static RagClient fromEnvironment() {
        String url = System.getenv().getOrDefault("RAG_URL", "http://rag-api:8000");
        return new RagClient(url);
    }

    public URI baseUri() {
        return baseUri;
    }

    /** Forwards a JSON body to POST /ask and returns the raw JSON response. */
    public String ask(String jsonBody) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("ask"))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();
        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new IOException("RAG service returned HTTP " + response.statusCode());
        }
        return response.body();
    }
}
