# syntax=docker/dockerfile:1.7

FROM eclipse-temurin:21-jdk AS builder

WORKDIR /workspace

COPY gradlew gradlew
COPY gradle gradle
COPY build.gradle.kts settings.gradle.kts gradle.properties ./

RUN sed -i 's/\r$//' ./gradlew && chmod +x ./gradlew

COPY src src

RUN --mount=type=cache,target=/root/.gradle ./gradlew --no-daemon bootJar -x test

FROM eclipse-temurin:21-jre

WORKDIR /app

COPY --from=builder /workspace/build/libs/*.jar app.jar

ENV SPRING_DATASOURCE_URL=jdbc:postgresql://orchestration-db:5432/orchestration_db
ENV SPRING_DATASOURCE_USERNAME=orch_user
ENV SPRING_DATASOURCE_PASSWORD=orch_pass
ENV FINGERPRINT_SERVICE_URL=http://fingerprint-service:8000
ENV JWT_SECRET=supersecretkeysupersecretkey
ENV JWT_EXPIRATION=86400000

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
