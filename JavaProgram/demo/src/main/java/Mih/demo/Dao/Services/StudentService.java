package Mih.demo.Dao.Services;

import Mih.demo.Modules.Student;

import java.util.List;

public interface StudentService {



    Student findStudentByNumber(String number);

    List<Student> getAllStudents();

    void createStudent(Student student);

    void createStudents(List<Student> students);

    void deleteStudentById(String studentId);
    
    void updateStudentById(Student student);

    public int updateStudent(Student student);

    List<Student> getBatchStudents();
}
