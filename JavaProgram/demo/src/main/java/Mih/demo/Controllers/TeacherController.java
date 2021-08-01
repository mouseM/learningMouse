package Mih.demo.Controllers;

import Mih.demo.Dao.Services.TeacherService;
import Mih.demo.Modules.Annotation;
import Mih.demo.Modules.Response;
import Mih.demo.Modules.Student;
import Mih.demo.Modules.Teacher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Controller
@RequestMapping("/teacher")
public class TeacherController {

    @Autowired
    TeacherService teacherService;

    @RequestMapping("/getteacherbyid")
    @ResponseBody
    public Response getTeacherById(@RequestParam("id")String number) {
        Response response = new Response();
        try {
            Teacher teacher = teacherService.findTeacherById(number);
            response.setState(200);
            response.setObj(teacher);
        } catch (Exception e) {
            response.setState(500);
        }
        return response;
    }

    @RequestMapping("/getallteachers")
    @ResponseBody
    public String getAllTeachers() {
        List<Teacher> teachers = teacherService.getAllTeachers();
        StringBuilder result = new StringBuilder();
        teachers.forEach(teacher -> {
            result.append(teacher.toString() + "\n");
        });
        return result.toString();
    }




    @RequestMapping(value = "/createteacher", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public Response createTeacher(@RequestBody Teacher teacher) {
        Response response = new Response();
        try {
            teacherService.createTeacher(teacher);
            response.setState(200);
        } catch (Exception e) {
            response.setState(500);
        }
        return response;
    }
}
