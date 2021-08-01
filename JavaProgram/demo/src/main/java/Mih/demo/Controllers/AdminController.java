package Mih.demo.Controllers;

import Mih.demo.Dao.Services.AdminService;
import Mih.demo.Modules.Annotation;
import Mih.demo.Modules.Response;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.Date;

@Controller
@RequestMapping("/admin")
public class AdminController {


    @Autowired
    AdminService adminService;

    @RequestMapping(value = "/createstudent", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public Response createStudent() {
        Response response = new Response();
        return response;
    }

    @RequestMapping(value = "/createteacher", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public Response createTeacher() {
        Response response = new Response();
        return response;
    }

    @RequestMapping(value = "/publishannotaions", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public Response publishAnnotations(@RequestBody Annotation annotation) {
        Date date = annotation.getDate();
        String detials = annotation.getDetials();
        adminService.publishAnnotation(annotation);
        Response response = new Response();
        return response;
    }
}
