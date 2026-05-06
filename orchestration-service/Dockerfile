FROM eclipse-temurin:21-jdk

WORKDIR /app

COPY build/libs/*.jar app.jar

ENV SPRING_DATASOURCE_URL=jdbc:postgresql://orchestration-db:5432/orchestration_db
ENV SPRING_DATASOURCE_USERNAME=orch_user
ENV SPRING_DATASOURCE_PASSWORD=orch_pass
ENV FINGERPRINT_SERVICE_URL=http://fingerprint-service:8000
ENV JWT_SECRET=supersecretkeysupersecretkey
ENV JWT_EXPIRATION=86400000
ENV SPRING_KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV SPRING_KAFKA_CONSUMER_GROUP_ID=orchestration-service
ENV ML_KAFKA_REQUEST_TOPIC=ml-recognition-requests
ENV ML_RESPONSE_TOPIC=ml-recognition-responses
ENV ML_TIMEOUT_MS=15000

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]