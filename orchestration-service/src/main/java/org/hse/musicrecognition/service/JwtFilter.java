package org.hse.musicrecognition.service;

import io.jsonwebtoken.io.IOException;
import org.hse.musicrecognition.domain.User;
import org.hse.musicrecognition.repository.UserRepository;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.util.List;

public class JwtFilter extends OncePerRequestFilter {

    private final JwtService jwtService;
    private final UserRepository userRepo;

    public JwtFilter(JwtService jwtService, UserRepository userRepo) {
        this.jwtService = jwtService;
        this.userRepo = userRepo;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws IOException, ServletException, java.io.IOException {

        String header = request.getHeader("Authorization");

        if (header == null || !header.startsWith("Bearer ")) {
            chain.doFilter(request, response);
            return;
        }

        String token = header.substring(7);
        String username = jwtService.extractUsername(token);

        User user = userRepo.findByUsername(username).orElse(null);
        if (user != null) {
            UsernamePasswordAuthenticationToken auth =
                    new UsernamePasswordAuthenticationToken(username, null,
                            List.of(new SimpleGrantedAuthority(user.getRole())));
            SecurityContextHolder.getContext().setAuthentication(auth);
        }

        chain.doFilter(request, response);
    }
}
