package org.hse.musicrecognition.service;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class JwtServiceTest {

    @Test
    void generatedTokenContainsUsername() {
        JwtService service = new JwtService();
        ReflectionTestUtils.setField(
                service,
                "secret",
                "0123456789012345678901234567890123456789012345678901234567891234"
        );
        ReflectionTestUtils.setField(service, "expiration", 60_000L);

        String token = service.generateToken("alice");

        assertThat(service.extractUsername(token)).isEqualTo("alice");
    }
}
