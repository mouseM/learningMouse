package Mih.demo.Dao;

import Mih.demo.Modules.Student;

import java.util.List;

public interface StudentService {



    Student findStudentByNumber(String number);

    List<Student> getAllStudents();

    void createStudent(Student student);

    void createStudents(List<Student> students);

    void deleteStudentById(String studentId);
}
