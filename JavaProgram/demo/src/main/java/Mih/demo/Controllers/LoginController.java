package Mih.demo.Controllers;

import Mih.demo.Dao.Services.UserService;
import Mih.demo.Modules.Response;
import Mih.demo.Modules.Annotation;
import Mih.demo.Modules.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.configurationprocessor.json.JSONObject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping("/login")
public class LoginController {

    @Autowired
    UserService userService;

    @RequestMapping(value = "/test", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public User test() {
        User user = new User();
        user.setPassWord("mihao0804");
        user.setUserName("mihao");
        return user;
    }

    @RequestMapping(value = "/doLogin", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public Response login(@RequestBody User user) {
        String userName = user.getUserName();
        String password = user.getPassWord();
        Response response = new Response();
        try {
            int userId = userService.getUserId(userName, password);
            user.setUserId(userId);
            response.setState(200);
            response.setObj(user);
        } catch (Exception e) {
            response.setState(500);
            response.setMessage("no such user!");
        }

        return response;
    }

    @RequestMapping(value = "/annotations", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public List<Annotation> getAnnotations() {
        List<Annotation> annotations = new ArrayList<>();
        Annotation annotation1 = new Annotation();
        annotation1.setDetials("annotation1");
        Annotation annotation2 = new Annotation();
        annotation2.setDetials("annotation2");
        annotations.add(annotation1);
        annotations.add(annotation2);
        return annotations;
    }
}
