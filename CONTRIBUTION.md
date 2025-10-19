# Contributing to CoffeCrawler

Hi there! Thank you for your interest in contributing to **CoffeCrawler** ☕. We genuinely appreciate your time and effort.

All contributions, whether it's bug reports, new feature requests, or code improvements, are warmly welcomed. This guide will help you understand the best way to contribute.

## 📜 Code of Conduct

To ensure a friendly and inclusive environment for everyone, we expect all contributors to read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). (Note: You may need to create this `CODE_OF_CONDUCT.md` file separately).

## 🚀 How to Contribute

We use the standard GitHub workflow (Fork & Pull Request). If you're new to this, don't worry, here are the steps.

### 1. Project Setup

Before you start, make sure you have set up your development environment according to the instructions in `README.md`.

1.  **Fork** this repository to your GitHub account.
2.  **Clone** your fork to your local machine:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/CoffeCrawler.git](https://github.com/YOUR_USERNAME/CoffeCrawler.git)
    cd CoffeCrawler
    ```
3.  Open the project in **Android Studio**.
4.  **Very Important:** Get your own Google Maps API Key from the Google Cloud Console.
5.  Create a `local.properties` file in the project's root directory (if it doesn't already exist).
6.  Add your API key to the `local.properties` file:
    ```properties
    MAPS_API_KEY=PASTE_YOUR_API_KEY_HERE
    ```
7.  Let Android Studio sync Gradle, then build the project to ensure everything is working smoothly.

### 2. Making Changes

1.  Create a **new branch** for your changes. Use a descriptive name, for example:
    * New feature: `git checkout -b feature/add-user-rating`
    * Bug fix: `git checkout -b bugfix/fix-detail-layout`
2.  Write your code!
3.  Make sure to follow the existing code standards (see below).
4.  **Commit** your changes with clear and descriptive messages.

### 3. Submitting a Pull Request (PR)

1.  **Push** your branch to your fork on GitHub:
    ```bash
    git push origin feature/your-feature-name
    ```
2.  Go to the CoffeCrawler repository on GitHub. You will see a button to **"Compare & pull request"**.
3.  Give your Pull Request a clear title.
4.  In the PR description, explain the changes you made. If it fixes an existing "Issue", link it using keywords like `Closes #123`.
5.  Submit your PR and wait for a review!

## 🐞 Reporting Bugs

* Check the **Issues** tab to see if the bug has already been reported.
* If not, open a new "Issue".
* Provide a clear and descriptive title.
* Explain the **steps to reproduce** the bug.
* Include *expected behavior* (what should happen) and *actual behavior* (what is actually happening).
* If possible, include screenshots or the Android version you are using.

## ✨ Suggesting New Features

* Open a new "Issue" and add the "enhancement" or "feature request" label.
* Describe the feature you would like and **why** it would be useful for CoffeCrawler users.
* Explain a clear use case or *user story*.

## 🎨 Code Standards

Based on the technology used in this project:

* **Language:** All code is written in **Python**.
* **Architecture:** Please follow the existing architectural pattern (MVVM).
* **UI:** Use **Jetpack Compose** for all new UI components.
* **Asynchronicity:** Use **Kotlin Coroutines** for background or network operations.
* **Dependency Injection:** Use **Hilt** to provide dependencies.
* **Networking:** Use **Ktor Client** for API calls.
* **Database:** Use **Room** for local data storage.
* **Formatting:** Follow the official [Android Kotlin style guides](https://developer.android.com/kotlin/style-guide).

Thank you once again for your contribution!

===
Local [Indonesian]
===

# Panduan Kontribusi untuk CoffeCrawler

Halo! Terima kasih telah tertarik untuk berkontribusi pada **CoffeCrawler** ☕. Kami sangat menghargai waktu dan usaha Anda.

Setiap kontribusi, baik itu laporan bug, permintaan fitur baru, atau perbaikan kode, sangat kami sambut. Panduan ini akan membantu Anda memahami cara terbaik untuk berkontribusi.

## 📜 Kode Etik

Untuk memastikan lingkungan yang ramah dan inklusif bagi semua, kami mengharapkan semua kontributor untuk membaca dan mematuhi [Kode Etik](CODE_OF_CONDUCT.md) kami. (Catatan: Anda mungkin perlu membuat file `CODE_OF_CONDUCT.md` ini secara terpisah).

## 🚀 Cara Berkontribusi

Kami menggunakan alur kerja standar GitHub (Fork & Pull Request). Jika Anda baru dalam hal ini, jangan khawatir, berikut langkah-langkahnya.

### 1. Pengaturan Proyek

Sebelum Anda mulai, pastikan Anda telah mengatur lingkungan pengembangan Anda sesuai dengan petunjuk di `README.md`.

1.  **Fork** repositori ini ke akun GitHub Anda.
2.  **Clone** fork Anda ke mesin lokal Anda:
    ```bash
    git clone [https://github.com/USERNAME_ANDA/CoffeCrawler.git](https://github.com/USERNAME_ANDA/CoffeCrawler.git)
    cd CoffeCrawler
    ```
3.  Buka proyek di **Android Studio**.
4.  **Sangat Penting:** Dapatkan Google Maps API Key Anda sendiri dari Google Cloud Console.
5.  Buat file `local.properties` di direktori root proyek (jika belum ada).
6.  Tambahkan API key Anda ke file `local.properties`:
    ```properties
    MAPS_API_KEY=MASUKKAN_API_KEY_ANDA_DI_SINI
    ```
7.  Biarkan Android Studio men-sinkronkan Gradle, lalu build proyek untuk memastikan semuanya berjalan lancar.

### 2. Membuat Perubahan

1.  Buat **branch baru** untuk perubahan Anda. Gunakan nama yang deskriptif, misalnya:
    * Fitur baru: `git checkout -b fitur/tambah-rating-pengguna`
    * Perbaikan bug: `git checkout -b bugfix/perbaiki-layout-detail`
2.  Tulis kode Anda!
3.  Pastikan untuk mengikuti standar kode yang ada (lihat di bawah).
4.  **Commit** perubahan Anda dengan pesan yang jelas dan deskriptif.

### 3. Mengajukan Pull Request (PR)

1.  **Push** branch Anda ke fork Anda di GitHub:
    ```bash
    git push origin fitur/nama-fitur-anda
    ```
2.  Buka repositori CoffeCrawler di GitHub. Anda akan melihat tombol untuk **"Compare & pull request"**.
3.  Beri judul yang jelas pada Pull Request Anda.
4.  Di deskripsi PR, jelaskan perubahan yang Anda buat. Jika ini memperbaiki "Issue" yang ada, tautkan dengan menggunakan kata kunci seperti `Closes #123`.
5.  Kirim PR Anda dan tunggu review!

## 🐞 Melaporkan Bug

* Periksa tab **Issues** untuk melihat apakah bug tersebut sudah dilaporkan.
* Jika belum, buka "Issue" baru.
* Beri judul yang jelas dan deskriptif.
* Jelaskan **langkah-langkah untuk mereproduksi** bug tersebut.
* Sertakan *expected behavior* (apa yang seharusnya terjadi) dan *actual behavior* (apa yang sebenarnya terjadi).
* Jika memungkinkan, sertakan tangkapan layar atau versi Android yang Anda gunakan.

## ✨ Menyarankan Fitur Baru

* Buka "Issue" baru dan berikan label "enhancement" atau "feature request".
* Jelaskan fitur yang Anda inginkan dan **mengapa** fitur tersebut akan berguna bagi pengguna CoffeCrawler.
* Jelaskan skenario penggunaan atau *user story* yang jelas.

## 🎨 Standar Kode

Berdasarkan teknologi yang digunakan dalam proyek ini:

* **Bahasa:** Seluruh kode ditulis dalam **Kotlin**.
* **Arsitektur:** Harap ikuti pola arsitektur yang ada (kemungkinan MVVM).
* **UI:** Gunakan **Jetpack Compose** untuk semua komponen UI baru.
* **Asinkron:** Gunakan **Kotlin Coroutines** untuk operasi latar belakang atau jaringan.
* **Dependency Injection:** Gunakan **Hilt** untuk menyediakan dependensi.
* **Networking:** Gunakan **Ktor Client** untuk panggilan API.
* **Database:** Gunakan **Room** untuk penyimpanan data lokal.
* **Formatting:** Ikuti panduan gaya Kotlin resmi dari Android.

Terima kasih sekali lagi atas kontribusi Anda!
