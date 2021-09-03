package Mih.demo.Dao;

import Mih.demo.CacheServer.RedisServer;
import Mih.demo.Dao.Services.StudentService;
import Mih.demo.Mappers.StudentMapper;
import Mih.demo.Modules.Student;
import com.github.pagehelper.Page;
import com.github.pagehelper.PageHelper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Date;
import java.util.List;

@Component
public class StudentImp implements StudentService {
    @Autowired
    StudentMapper studentMapper;

    @Autowired
    RedisServer redisServer;

    @Autowired
    CacheManager cacheManager;

    /*
    对于查询操作首先从缓存中进行查询
     */
    @Transactional
    @Cacheable(value = "student", keyGenerator = "simpleKeyGenerator", cacheManager = "cacheManager")
    @Override
    public Student findStudentByNumber(String number) {
        Student student = studentMapper.getStudentByNumber(number);
        return student;
    }

    @Transactional
    @Override
    public List<Student> getAllStudents() {
        return studentMapper.getAllStudents();
    }

    /*
    对于创建操作， 直接修改数据库
     */
    @Transactional(propagation = Propagation.REQUIRED)
    @Override
    public void createStudent(Student student) {
        studentMapper.createStudent(student);
    }

    /*
    对于创建操作， 直接修改数据库
     */
    @Transactional
    @Override
    public void createStudents(List<Student> students) {
        studentMapper.createStudents(students);
    }

    @Transactional
    @Override
    @CacheEvict(value = "student", key = "#studentId")
    public void deleteStudentById(String studentId) {
        studentMapper.delStudentById(studentId);
    }

    @Transactional
    @Override
    public void updateStudentById(Student student) {
        int number = student.getStudentId();
        String name = student.getName();
        Date birthday = student.getBirthday();
        String sex = student.getSex();
        String telephoneNumber = student.getTelephoneNumber();
        String e_mailAddress = student.getE_mailAddress();
        String address = student.getAddress();
        studentMapper.updateStudentById(
                number, name, birthday, sex,
                telephoneNumber, e_mailAddress, address);
    }

    @Override
    @Transactional
    public int updateStudent(Student student) {
        return studentMapper.updateStudent(student);
    }

    @Transactional
    @Override
    public List<Student> getBatchStudents() {
        int pageNumber = 2;
        int pageSize = 5;
        Page page = PageHelper.startPage(pageNumber, pageSize);
        List<Student> students = studentMapper.getBatchStudents();
        return students;
    }

}
