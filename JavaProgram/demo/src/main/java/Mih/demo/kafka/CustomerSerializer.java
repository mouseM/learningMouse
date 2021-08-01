package Mih.demo.kafka;


import org.apache.kafka.common.header.Headers;
import org.apache.kafka.common.serialization.Serializer;
import org.springframework.boot.configurationprocessor.json.JSONObject;

import java.util.Map;

public class CustomerSerializer implements Serializer<JSONObject> {
    @Override
    public void configure(Map<String, ?> configs, boolean isKey) {
        //TODO
    }

    @Override
    public byte[] serialize(String topic, JSONObject data) {
//        return JSONObjecttoJSONBytes(data);
        return null;
    }

    @Override
    public byte[] serialize(String topic, Headers headers, JSONObject data) {
        return new byte[0];
    }

    @Override
    public void close() {
        //TODO
    }
}
