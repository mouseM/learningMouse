package Mih.demo.Controllers;

import Mih.demo.Dao.Services.UserService;
import Mih.demo.Modules.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@Controller
@RequestMapping("/user")
public class UserController {

    @Autowired
    UserService userService;

    @RequestMapping(value = "/createuser", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public void createUser(@RequestBody User user) {

        userService.createUser(user);
    }

    @RequestMapping(value = "/getuserid", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public int getUserId(@RequestParam("username") String userName,
                          @RequestParam("password") String passWord) {

        return userService.getUserId(userName, passWord);
    }

}
