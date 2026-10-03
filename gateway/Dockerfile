# Build the WAR, then run it on Open Liberty (Jakarta EE 10).
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /src
COPY pom.xml .
RUN mvn -q -B dependency:go-offline
COPY src ./src
RUN mvn -q -B package

FROM icr.io/appcafe/open-liberty:kernel-slim-java17-openj9-ubi
COPY --chown=1001:0 src/main/liberty/config/server.xml /config/
RUN features.sh
COPY --chown=1001:0 --from=build /src/target/plm-gateway.war /config/apps/
EXPOSE 9080
