package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.User;
import org.hse.musicrecognition.dto.AuthRequest;
import org.hse.musicrecognition.dto.AuthResponse;
import org.hse.musicrecognition.exception.BadRequestException;
import org.hse.musicrecognition.exception.UnauthorizedException;
import org.hse.musicrecognition.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthResponse register(AuthRequest request) {
        if (userRepository.existsByUsername(request.username())) {
            throw new BadRequestException("Username is already taken");
        }

        User user = User.builder()
                .username(request.username())
                .passwordHash(passwordEncoder.encode(request.password()))
                .role("ROLE_USER")
                .build();

        userRepository.save(user);

        return createAuthResponse(user);
    }

    public AuthResponse login(AuthRequest request) {
        User user = userRepository.findByUsername(request.username())
                .orElseThrow(() -> new UnauthorizedException("Invalid username or password"));

        if (!passwordEncoder.matches(request.password(), user.getPasswordHash())) {
            throw new UnauthorizedException("Invalid username or password");
        }

        return createAuthResponse(user);
    }

    private AuthResponse createAuthResponse(User user) {
        String token = jwtService.generateToken(user.getUsername());
        return new AuthResponse(token, "Bearer");
    }
}