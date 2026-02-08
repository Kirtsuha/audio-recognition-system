plugins {
    id("java")
    id("org.springframework.boot") version "3.2.5"
}

group = "org.hse"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-webflux:3.2.12")

    // === SWAGGER для Boot 3 + WebFlux ===
    implementation("org.springdoc:springdoc-openapi-starter-webflux-ui:2.5.0")

    // Lombok
    compileOnly("org.projectlombok:lombok:1.18.32")
    annotationProcessor("org.projectlombok:lombok:1.18.32")

    // Tests
    testImplementation("org.springframework.boot:spring-boot-starter-test:3.2.12")
    testImplementation("io.projectreactor:reactor-test:3.8.2")
}

tasks.test {
    useJUnitPlatform()
}