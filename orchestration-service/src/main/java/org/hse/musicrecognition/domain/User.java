package org.hse.musicrecognition.domain;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "users")
@Data
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {
    @Id
    @GeneratedValue
    private Long id;

    @Column(unique = true)
    private String username;

    private String passwordHash;

    @Column
    private String role = "ROLE_USER";

    public String getRole() {
        return role;
    }

    public Long getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getPasswordHash() {
        return passwordHash;
    }
}
