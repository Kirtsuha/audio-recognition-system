package org.hse.musicrecognition.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@ConfigurationProperties(prefix = "ml")
@Component
@Getter
@Setter
public class MlProperties {

    private long timeoutMs;
    private Kafka kafka;

    @Getter
    @Setter
    public static class Kafka {
        private String requestTopic;
        private String responseTopic;
    }
}
