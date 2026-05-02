package org.hse.musicrecognition.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.AuthRequest;
import org.hse.musicrecognition.dto.AuthResponse;
import org.hse.musicrecognition.service.AuthService;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public AuthResponse register(@Valid @RequestBody AuthRequest request) {
        return authService.register(request);
    }

    @PostMapping("/login")
    public AuthResponse login(@Valid @RequestBody AuthRequest request) {
        return authService.login(request);
    }
}