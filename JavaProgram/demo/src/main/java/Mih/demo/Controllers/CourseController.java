package Mih.demo.Controllers;

import Mih.demo.Dao.Services.ClassService;
import Mih.demo.Modules.Course;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.List;

@Controller
@RequestMapping("/course")
public class CourseController {

    @Autowired
    ClassService classService;

    @Autowired
    RestfulTest restfulTest;

    @RequestMapping(value = "/getcoursesbyteacherid", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public List<Course> getCoursesByTeacherId(@RequestParam("tid") String teacherId) {
        return classService.getCoursesByTeacherId(teacherId);
    }
}
