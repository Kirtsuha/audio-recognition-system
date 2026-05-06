package org.hse.musicrecognition.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "recognition")
public class RecognitionProperties {

    private Fingerprint fingerprint = new Fingerprint();
    private Storage storage = new Storage();

    @Getter
    @Setter
    public static class Fingerprint {
        private int topK = 20;
    }

    @Getter
    @Setter
    public static class Storage {
        private String queryAudioBucket = "recognition-queries";
        private String queryAudioPrefix = "debug";
        private String referenceAudioBucket = "full-tracks";
    }
}