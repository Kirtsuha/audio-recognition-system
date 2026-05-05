package org.hse.musicrecognition.config;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;
import org.hse.musicrecognition.dto.MlRecognitionRequest;
import org.hse.musicrecognition.dto.MlRecognitionResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.core.ConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaProducerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.core.ProducerFactory;
import org.springframework.kafka.support.serializer.JsonDeserializer;
import org.springframework.kafka.support.serializer.JsonSerializer;

import java.util.HashMap;
import java.util.Map;

@Configuration
@EnableKafka
public class KafkaConfig {

    @Bean
    public ProducerFactory<String, MlRecognitionRequest> mlRequestProducerFactory(
            @Value("${spring.kafka.bootstrap-servers}") String bootstrapServers,
            @Value("${spring.kafka.producer.properties.max.request.size:10485760}") int maxRequestSize,
            @Value("${spring.kafka.producer.properties.buffer.memory:33554432}") long bufferMemory
    ) {
        Map<String, Object> config = new HashMap<>();
        config.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        config.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        config.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        config.put(JsonSerializer.ADD_TYPE_INFO_HEADERS, false);

        config.put(ProducerConfig.MAX_REQUEST_SIZE_CONFIG, maxRequestSize);
        config.put(ProducerConfig.BUFFER_MEMORY_CONFIG, bufferMemory);

        return new DefaultKafkaProducerFactory<>(config);
    }

    @Bean
    public KafkaTemplate<String, MlRecognitionRequest> mlRequestKafkaTemplate(
            ProducerFactory<String, MlRecognitionRequest> mlRequestProducerFactory
    ) {
        return new KafkaTemplate<>(mlRequestProducerFactory);
    }

    @Bean
    public ConsumerFactory<String, MlRecognitionResponse> mlResponseConsumerFactory(
            @Value("${spring.kafka.bootstrap-servers}") String bootstrapServers,
            @Value("${spring.kafka.consumer.group-id}") String groupId
    ) {
        JsonDeserializer<MlRecognitionResponse> valueDeserializer =
                new JsonDeserializer<>(MlRecognitionResponse.class);
        valueDeserializer.addTrustedPackages("org.hse.musicrecognition.dto");
        valueDeserializer.ignoreTypeHeaders();

        Map<String, Object> config = new HashMap<>();
        config.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        config.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        config.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        config.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        // config.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        // config.put(JsonDeserializer.TRUSTED_PACKAGES, "org.hse.musicrecognition.dto");

        return new DefaultKafkaConsumerFactory<>(config, new StringDeserializer(), valueDeserializer);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, MlRecognitionResponse> kafkaListenerContainerFactory(
            ConsumerFactory<String, MlRecognitionResponse> mlResponseConsumerFactory
    ) {
        ConcurrentKafkaListenerContainerFactory<String, MlRecognitionResponse> factory =
                new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(mlResponseConsumerFactory);
        return factory;
    }
}
