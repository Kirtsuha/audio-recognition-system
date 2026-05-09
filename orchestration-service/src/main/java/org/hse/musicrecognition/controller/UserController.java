package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.UserResponse;
import org.hse.musicrecognition.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.security.Principal;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/me")
    public UserResponse me(Principal principal) {
        return userService.getCurrentUser(principal.getName());
    }

    @DeleteMapping("/me")
    public ResponseEntity<Void> deleteMe(Principal principal) {
        userService.deleteOwnAccount(principal.getName());
        return ResponseEntity.noContent().build();
    }
}