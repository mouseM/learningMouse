package Mih.demo.Controllers;

import Mih.demo.Dao.TestService;
import Mih.demo.Modules.Student;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
//@Lazy
public class RestfulTest {

    Logger logger = LoggerFactory.getLogger(RestfulTest.class);

    @Autowired
    TestService service2;

    @GetMapping("/hello")
    public String hello() {
//        logger.debug("Request in /hello");
        return "hello World";
    }

    @PostMapping("/test")
    public String test(@RequestBody Student student) {
//        logger.debug("Request in /hello");
        service2.test(student);
        return "end!";
    }


}
