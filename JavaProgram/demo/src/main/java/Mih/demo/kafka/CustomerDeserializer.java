package Mih.demo.kafka;


import org.apache.kafka.common.header.Headers;
import org.apache.kafka.common.serialization.Deserializer;
import org.springframework.boot.configurationprocessor.json.JSONObject;

import java.util.Map;

public class CustomerDeserializer implements Deserializer<JSONObject> {
    @Override
    public void configure(Map<String, ?> configs, boolean isKey) {
        //TODO
    }

    @Override
    public JSONObject deserialize(String topic, byte[] data) {
//        return JSON(data, JSONObject.class);
        return null;
    }

    @Override
    public JSONObject deserialize(String topic, Headers headers, byte[] data) {
        return null;
    }


    @Override
    public void close() {
        //TODO
    }
}
