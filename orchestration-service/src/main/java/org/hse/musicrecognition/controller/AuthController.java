package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.User;
import org.hse.musicrecognition.dto.AuthRequest;
import org.hse.musicrecognition.repository.UserRepository;
import org.hse.musicrecognition.service.JwtService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final UserRepository userRepo;
    private final PasswordEncoder encoder;
    private final JwtService jwtService;

    @PostMapping("/register")
    public String register(@RequestBody AuthRequest req) {
        User user = User.builder()
                .username(req.username())
                .passwordHash(encoder.encode(req.password()))
                .build();
        userRepo.save(user);
        return "Registered";
    }

    @PostMapping("/login")
    public String login(@RequestBody AuthRequest req) {
        User user = userRepo.findByUsername(req.username()).orElseThrow();
        if (!encoder.matches(req.password(), user.getPasswordHash()))
            throw new RuntimeException("Invalid credentials");

        return jwtService.generateToken(user.getUsername());
    }
}

