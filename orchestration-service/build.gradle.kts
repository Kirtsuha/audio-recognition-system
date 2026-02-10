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

    implementation("org.springframework.security:spring-security-web:7.0.0")

    // === SWAGGER для Boot 3 + WebFlux ===
    implementation("org.springdoc:springdoc-openapi-starter-webflux-ui:2.5.0")

    implementation("org.hibernate.orm:hibernate-core:7.2.2.Final")

    // Lombok
    compileOnly("org.projectlombok:lombok:1.18.32")
    annotationProcessor("org.projectlombok:lombok:1.18.32")

    // Tests
    testImplementation("org.springframework.boot:spring-boot-starter-test:3.2.12")
    testImplementation("io.projectreactor:reactor-test:3.8.2")

    implementation("org.springframework.boot:spring-boot-starter-security:3.2.12")
    implementation("org.springframework.boot:spring-boot-starter-data-jpa:3.2.12")
    implementation("org.springframework.boot:spring-boot-starter-oauth2-resource-server:3.2.12")

    implementation("org.postgresql:postgresql:42.1.4")

    implementation("io.jsonwebtoken:jjwt-api:0.12.5")
    runtimeOnly("io.jsonwebtoken:jjwt-impl:0.12.5")
    runtimeOnly("io.jsonwebtoken:jjwt-jackson:0.12.5")
}

tasks.test {
    useJUnitPlatform()
}