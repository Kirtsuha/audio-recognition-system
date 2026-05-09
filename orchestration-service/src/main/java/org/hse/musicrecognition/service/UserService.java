package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.User;
import org.hse.musicrecognition.dto.UserResponse;
import org.hse.musicrecognition.exception.NotFoundException;
import org.hse.musicrecognition.repository.HistoryRepository;
import org.hse.musicrecognition.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final HistoryRepository historyRepository;

    public UserResponse getCurrentUser(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new NotFoundException("User not found: " + username));

        return new UserResponse(
                user.getId(),
                user.getUsername(),
                user.getRole()
        );
    }

    @Transactional
    public void deleteOwnAccount(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new NotFoundException("User not found: " + username));

        historyRepository.deleteByUserId(user.getId());
        userRepository.delete(user);
    }
}