package Mih.demo.Modules;

import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.validation.constraints.*;
import java.io.Serializable;
import java.util.Date;

public class Student implements Serializable {

    @NotNull(message = "学生ID不能为空")
    @Min(1)
    private int studentId;

    @NotNull(message = "学生姓名不能为空")
    @NotEmpty(message = "学生姓名不能为空串")
    private String name;

    @NotNull(message = "出生日期不能为空")
    @Past(message = "出生日期输入错误")
    private Date birthday;

    private String sex;

    @NotNull(message = "电话号码不能为空")
    @Pattern(regexp = "1[0-9]{10}", message = "电话号码输入格式错误")
    private String telephoneNumber;

    @NotNull(message = "电子邮箱输入不能为空")
    @Email(message = "电子邮箱输入格式错误")
    private String e_mailAddress;

    private String address;

    public Student() {
    }

    public int getStudentId() {
        return studentId;
    }

    public void setStudentId(int studentId) {
        this.studentId = studentId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Date getBirthday() {
        return birthday;
    }

    public void setBirthday(Date birthday) {
        this.birthday = birthday;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    @Override
    public String toString() {
        return "Student{" +
                "studentId=" + studentId +
                ", name='" + name + '\'' +
                ", birthday=" + birthday +
                ", sex='" + sex + '\'' +
                ", telephoneNumber='" + telephoneNumber + '\'' +
                ", e_mailAddress='" + e_mailAddress + '\'' +
                ", address='" + address + '\'' +
                '}';
    }

    public String getTelephoneNumber() {
        return telephoneNumber;
    }

    public void setTelephoneNumber(String telephoneNumber) {
        this.telephoneNumber = telephoneNumber;
    }

    public String getE_mailAddress() {
        return e_mailAddress;
    }

    public void setE_mailAddress(String e_mailAddress) {
        this.e_mailAddress = e_mailAddress;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }
}
