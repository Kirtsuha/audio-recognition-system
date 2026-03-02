FROM gradle:8.7-jdk21 AS build
WORKDIR /build

COPY . .

RUN sed -i 's/\r$//' gradlew
RUN chmod +x gradlew
RUN ./gradlew bootJar -x test

FROM eclipse-temurin:21-jdk
WORKDIR /app
COPY --from=build /build/build/libs/*.jar app.jar


ENV SPRING_DATASOURCE_URL=jdbc:postgresql://orchestration-db:5432/orchestration_db
ENV SPRING_DATASOURCE_USERNAME=orch_user
ENV SPRING_DATASOURCE_PASSWORD=orch_pass
ENV FINGERPRINT_SERVICE_URL=http://fingerprint-service:8000/index
ENV JWT_SECRET=supersecretkeysupersecretkey
ENV JWT_EXPIRATION=86400000

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
