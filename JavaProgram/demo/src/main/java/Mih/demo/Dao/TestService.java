package Mih.demo.Dao;

import Mih.demo.Dao.Services.StudentService;
import Mih.demo.Dao.Services.TeacherService;
import Mih.demo.Modules.Student;
import Mih.demo.Modules.Teacher;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

@Service
public class TestService {

    @Autowired
    TeacherService teacherService;

    @Autowired
    StudentService studentService;

    @Transactional(propagation = Propagation.REQUIRED)
    public void test(Student student) {
//        try {
            studentService.createStudent(student);
            Teacher teacher = new Teacher();
            teacher.setTeacherId(26);
            teacher.setName("tmih");
            teacher.setSex("b");
            teacher.setBirthday("2033");
            teacher.setTelephoneNumber("1234");
            teacher.setAddress("china");
            teacher.setE_mailAddress("1234@qq.com");
        try {
            teacherService.createTeacher(teacher);
        } catch (Exception e) {
            e.printStackTrace();
        }
//            teacherService.createTeacher(teacher);

//        } catch (Exception e) {
//
//        }
    }
}
