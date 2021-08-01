package Mih.demo.Controllers;

import Mih.demo.Dao.Services.ClassService;
import Mih.demo.Dao.Services.ScoreService;
import Mih.demo.Dao.Services.StudentService;
import Mih.demo.Modules.Course;
import Mih.demo.Modules.Response;
import Mih.demo.Modules.Student;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;


@RestController
public class StudentController {

    @Autowired
    StudentService studentService;

    @Autowired
    ScoreService scoreService;

    @Autowired
    ClassService classService;

    @RequestMapping("/getstudentbyid")
    @ResponseBody
    public Response getStudentByNumber(@RequestParam("id") String number) {
        Response response = new Response();
        try {
            Student student = studentService.findStudentByNumber(number);
            response.setObj(student);
            response.setState(200);
        } catch (Exception e) {
            response.setState(500);
            response.setMessage("error in find student by id!");
        }

        return response;
    }

    @RequestMapping("/getallstudents")
    @ResponseBody
    public String getAllStudents() {
        List<Student> students = studentService.getAllStudents();
        StringBuilder result = new StringBuilder();
        students.forEach(student -> {
            result.append(student.toString() + "/n");
        });
        return result.toString();
    }

    @PostMapping("/createstudent")
    public Response createStudent(@RequestBody @Validated Student student) {
        Response response = new Response();
        try {
            studentService.createStudent(student);
            response.setState(200);
        } catch (Exception e) {
            response.setState(500);
        }
        return response;
    }

    @RequestMapping(value = "/createstudents", method = RequestMethod.POST, produces = "application/json;charset=UTF-8")
    @ResponseBody
    public void createStudents(@RequestBody @Validated List<Student> students) {
        studentService.createStudents(students);
    }

    @RequestMapping(value = "/deletestudentbyid", method = RequestMethod.DELETE)
    @ResponseBody
    public void deleteStudentById(@RequestParam("id") String number) {
        studentService.deleteStudentById(number);
    }

    @RequestMapping(value = "/updatestudentbyid", method = RequestMethod.PUT)
    @ResponseBody
    public void updateStudent(@RequestParam("id") String number,Student student) {
        studentService.updateStudentById(student);
    }

    @RequestMapping(value = "/selectcourse", method = RequestMethod.POST)
    @ResponseBody
    public Response selectCourse(@RequestBody Course course) {
        classService.selectCourse(course);
        Response response = new Response();
        return response;
    }

    @PutMapping("/updatestudent")
    public Response updateStudent(@RequestBody @Validated Student student) {
        int updateSucessed = studentService.updateStudent(student);
        Response response = new Response();
        if (updateSucessed > 0) {
            response.setState(200);
            response.setMessage("更新成功！");
        } else {
            response.setState(500);
            response.setMessage("更新失败！");
        }
        return response;
    }

}
