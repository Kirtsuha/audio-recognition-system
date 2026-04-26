plugins {
    id("java")
    id("org.springframework.boot") version "3.2.5"
    id("io.spring.dependency-management") version "1.1.7"
}

group = "org.hse"
version = "1.0-SNAPSHOT"

java {
    sourceCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

dependencies {

    // === WEB (MVC) ===
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.kafka:spring-kafka")

    // === SECURITY ===
    implementation("org.springframework.boot:spring-boot-starter-security")

    // === JPA ===
    implementation("org.springframework.boot:spring-boot-starter-data-jpa")

    // === DATABASE ===
    runtimeOnly("org.postgresql:postgresql")

    // === SWAGGER (для MVC!) ===
    implementation("org.springdoc:springdoc-openapi-starter-webmvc-ui:2.5.0")

    // === JWT ===
    implementation("io.jsonwebtoken:jjwt-api:0.12.5")
    runtimeOnly("io.jsonwebtoken:jjwt-impl:0.12.5")
    runtimeOnly("io.jsonwebtoken:jjwt-jackson:0.12.5")

    // === Lombok ===
    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")

    // === TESTS ===
    testImplementation("org.springframework.boot:spring-boot-starter-test")
    testImplementation("org.springframework.kafka:spring-kafka-test")
}

tasks.test {
    useJUnitPlatform()
}
