package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.*;
import org.hse.musicrecognition.exception.BadRequestException;
import org.hse.musicrecognition.service.CatalogAdminService;
import org.springframework.http.MediaType;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.security.Principal;
import java.util.UUID;

@RestController
@RequestMapping("/api/admin")
@PreAuthorize("hasRole('ADMIN')")
@RequiredArgsConstructor
public class AdminCatalogController {

    private final CatalogAdminService catalogAdminService;

    @PostMapping(value = "/tracks", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public AdminTrackUploadResponse uploadTrack(
            @RequestPart("file") MultipartFile file,
            @RequestParam("title") String title,
            @RequestParam("artist") String artist,
            @RequestParam(value = "album", required = false) String album,
            Principal principal
    ) {
        return catalogAdminService.uploadTrack(
                principal.getName(),
                file,
                title,
                artist,
                album
        );
    }

    @PostMapping(value = "/tracks/bulk", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public AdminBulkUploadResponse uploadBulk(
            @RequestPart("file") MultipartFile file,
            Principal principal
    ) {
        if (file == null || file.isEmpty()) {
            throw new BadRequestException("Archive file is empty");
        }

        return catalogAdminService.startBulkImport(principal.getName());
    }

    @PostMapping("/catalog/sync")
    public CatalogSyncResponse syncCatalog(Principal principal) {
        return catalogAdminService.runIncrementalSync(principal.getName());
    }

    @PostMapping("/catalog/full-rebuild")
    public CatalogJobStartedResponse fullRebuild(Principal principal) {
        return catalogAdminService.startFullRebuild(principal.getName());
    }

    @GetMapping("/jobs/{jobId}")
    public CatalogJobResponse getJob(@PathVariable UUID jobId) {
        return catalogAdminService.getJob(jobId);
    }
}
