package Mih.demo.Mappers;

import Mih.demo.Modules.Student;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.Date;
import java.util.List;

@Mapper
public interface StudentMapper {

    Student getStudentByNumber(@Param("SId") String number);

    List<Student> getAllStudents();

    List<Student> getBatchStudents();

//    void createStudent(@Param("SId") int number,
//                       @Param("Sname") String name,
//                       @Param("Sage") Date birthday,
//                       @Param("Ssex") String sex,
//                       @Param("telephoneNumber") String telephoneNumber,
//                       @Param("e_mailAddress") String e_mailAddress,
//                       @Param("address") String address);

    void createStudent(Student student);

    void createStudents(@Param("students") List<Student> students);

    void delStudentById(@Param("SId") String studentId);

    void updateStudentById(@Param("SId") int number,
                           @Param("Sname") String name,
                           @Param("Sage") Date birthday,
                           @Param("Ssex") String sex,
                           @Param("telephoneNumber") String telephoneNumber,
                           @Param("e_mailAddress") String e_mailAddress,
                           @Param("address") String address);

    int updateStudent(Student student);
}
