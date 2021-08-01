package Mih.demo.Controllers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.*;

@RestController
//@Lazy
public class RestfulTest {

    Logger logger = LoggerFactory.getLogger(RestfulTest.class);

    @GetMapping("/hello")
    public String hello() {
//        logger.debug("Request in /hello");
        return "hello World";
    }

    @GetMapping("/test")
    public String test() {
//        logger.debug("Request in /hello");
        System.out.println(hello());
        return "hello test";
    }


}
