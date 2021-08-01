package Mih.demo.Controllers;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;


@RestController
public class KafkaTest {

    @Autowired
    private KafkaTemplate<String, Object> kafkaTemplate;

    @GetMapping("kafka/test")
    public void sendMessage(@RequestParam("message") String message) {
        kafkaTemplate.send("topic1", message);
    }
}
