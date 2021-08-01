package Mih.demo.Config;

import com.alibaba.fastjson.serializer.SerializerFeature;
import com.alibaba.fastjson.support.config.FastJsonConfig;
import com.alibaba.fastjson.support.spring.FastJsonHttpMessageConverter;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.util.List;
@Configuration
public class JsonConverterConfig implements WebMvcConfigurer {
    @Override
    public void extendMessageConverters(List<HttpMessageConverter<?>> converters) {
        FastJsonHttpMessageConverter fjc = new FastJsonHttpMessageConverter();
        FastJsonConfig fj = new FastJsonConfig();

        //字符类型字段如果为null，则输出"",而非null
//        fj.setSerializerFeatures(SerializerFeature.WriteNullStringAsEmpty);
        fj.setSerializerFeatures(SerializerFeature.WriteMapNullValue);
        fjc.setFastJsonConfig(fj);
        converters.add(fjc);
    }
}
