package Mih.demo.Controllers;

import Mih.demo.Dao.Services.ClassService;
import Mih.demo.Dao.Services.ScoreService;
import Mih.demo.Dao.Services.StudentService;
import Mih.demo.Mappers.StudentMapper;
import Mih.demo.Modules.Course;
import Mih.demo.Modules.Response;
import Mih.demo.Modules.Student;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;


@RestController
public class StudentController {

    @Autowired
    StudentService studentService;

    @Autowired
    ScoreService scoreService;

    @Autowired
    ClassService classService;

    @Autowired
    SqlSessionFactory sqlSessionFactory;

    @RequestMapping("/getstudentbyid")
    @ResponseBody
    public Response getStudentByNumber(@RequestParam("id") String number) {
        Response response = new Response();
        try {
            SqlSession getSession = sqlSessionFactory.openSession();
            SqlSession updateSession = sqlSessionFactory.openSession();
            StudentMapper studentMapper = getSession.getMapper(StudentMapper.class);
            StudentMapper updateMapper = updateSession.getMapper(StudentMapper.class);
            Student student = studentMapper.getStudentByNumber(number);
            Student newStudent = new Student();
            newStudent.setStudentId(6);
            newStudent.setName("miWuhao");
            updateMapper.updateStudent(newStudent);
            student = studentMapper.getStudentByNumber(number);
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

    @GetMapping(value = "/slectstudentsbatch")
    public Response slectStudentsBatch() {
        Response response = new Response();
        List<Student> batchStudents = studentService.getBatchStudents();
        if (batchStudents.size() > 0) {
            response.setState(200);
            response.setObj(batchStudents);
            response.setMessage("查询成功！");
        } else {
            response.setState(500);
            response.setMessage("查询失败！");
        }
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

    // 测试一：多线程向数据库中同一张表插入数据
    private static final AtomicInteger seqNum = new AtomicInteger(1);
    private static int BATCH = 1000;
    @GetMapping("/testcreatemanystudents")
    public Response testCreateManyStudents() {
        // 创建100,0000学生，分10条线程插入

        IntStream.rangeClosed(1, 10).forEach(i -> {
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    System.out.println(Thread.currentThread().getName() + " start create students!");
                    long startTime = System.currentTimeMillis();
                    List<Student> students = new ArrayList<>();
                    IntStream.rangeClosed(1, BATCH).forEach(i -> {
                        students.add(new Student());
                    });
                    fillStudent(students);
                    studentService.testCreateManyStudents(students);
                    System.out.println(Thread.currentThread().getName() + " end create students!");
                    long endTime = System.currentTimeMillis();
                    System.out.println("used time : " + (((double) endTime - (double) startTime)) / 1000);
                }
            }, "CreateStudentThread-" + i);
            thread.start();
        });
        Response response = new Response();
        response.setState(200);
        return response;
    }

    private void fillStudent(List<Student> students) {
        fillStudentId(students);
        fillStudentsBirthday(students);
        fillStudentName(students);
        fillStudentSex(students);
        fillStudentPhoneNumber(students);
        fillStudentEmailAddress(students);
        fillStudentAddress(students);
    }

    private static final Random random = new Random(6l);

    private void fillStudentId(List<Student> students) {
        students.forEach(student -> {
            student.setStudentId(seqNum.getAndIncrement());
        });
    }

    private void fillStudentsBirthday(List<Student> students) {
        int min = 1985;
        int max = 2010;

        students.forEach(student -> {
            int randomYear = random.nextInt(max - min) + min;
            int randomMouth = random.nextInt(12) + 1;
            int randomDay = random.nextInt(28) + 1;
            String mount = "";
            String day = "";
            if (randomMouth < 10) {
                mount =  "0" + randomMouth;
            } else {
                mount = mount + randomMouth;
            }
            if (randomDay < 10) {
                day = "0" + randomDay;
            } else {
                day = day + randomDay;
            }
            String birthday = randomYear + mount + day;
            student.setAddress(birthday);
        });
    }

    private void fillStudentSex(List<Student> students) {
        String[] sexs = new String[] {"M", "W"};
        students.forEach(student -> {
            int index = random.nextInt(2);
            student.setSex(sexs[index]);
        });
    }

    // 生成随机的名字
    private void fillStudentName(List<Student> students) {
        students.forEach(student -> {
            StringBuilder result = new StringBuilder("Mi ");
            // 最少3位，最多8位
            int length = random.nextInt(6) + 2;
            result.append((char) ('A' + random.nextInt(26)));
            IntStream.rangeClosed(1, length).forEach(i -> {
                result.append((char) ('a' + random.nextInt(26)));
            });
            student.setName(result.toString());
        });
    }

    private void fillStudentPhoneNumber(List<Student> students) {
        students.forEach(student -> {
            StringBuilder result = new StringBuilder("1");
            // 生成10位电话号码
            IntStream.rangeClosed(1, 10).forEach(index -> {
                int randomNum = random.nextInt(10);
                result.append(randomNum);
            });
            student.setTelephoneNumber(result.toString());
        });
    }

    // 生成QQ邮箱，随机生成，假设QQ号都是10位
    private void fillStudentEmailAddress(List<Student> students) {
        students.forEach(student -> {
            StringBuilder result = new StringBuilder();
            IntStream.rangeClosed(1, 10).forEach(i -> {
                int num = random.nextInt(10);
                result.append(num);
            });
            result.append("@qq.com");
            student.setE_mailAddress(result.toString());
        });
    }

    private void fillStudentAddress(List<Student> students) {
        String[] address = new String[] {"CN", "JP", "KR", "US", "UK", "AUS"};
        students.forEach(student -> {
            int index = random.nextInt(6);
            student.setAddress(address[index]);
        });
    }
}
