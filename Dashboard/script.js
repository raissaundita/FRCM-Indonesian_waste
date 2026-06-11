// =====================================================
// SCRIPT UTAMA DASHBOARD
// Membaca CSV per tahun, mengisi tabel, dan menampilkan
// popup detail provinsi dengan timeline dot saat diklik
// =====================================================

// Tahun yang sedang aktif
let tahunAktif = 2022;

// Tempat menyimpan semua data setelah CSV dibaca
// Contoh: semuaData[2019] = [ {provinsi, klaster, ...}, ... ]
const semuaData = {};

// Warna setiap klaster (label 1-4)
const WARNA_KLASTER = {
  1: "#29B6F6",  // biru
  2: "#66BB6A",  // hijau
  3: "#FBC02D",  // kuning
  4: "#EF5350",  // merah
};

const TAHUN_LIST  = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
const TAHUN_LABEL = TAHUN_LIST.map(String);

// =====================================================
// FUNGSI: Baca satu file CSV
// =====================================================
async function bacaCSV(tahun) {
  // Ambil file dari folder data/
  const response = await fetch(`data/klaster_${tahun}.csv`);
  const teks     = await response.text();

  // Pisahkan per baris
  const baris  = teks.trim().split("\n");
  // Baris pertama = header; hapus BOM (karakter tersembunyi) kalau ada
  const header = baris[0].replace(/^\uFEFF/, "").split(",");

  // Konversi setiap baris menjadi objek
  return baris.slice(1).map(b => {
    const kolom = b.split(",");
    const obj   = {};
    header.forEach((h, i) => obj[h.trim()] = kolom[i]?.trim());

    return {
      provinsi    : obj["Provinsi"],
      // label di CSV adalah 0-3 (indeks), kita tambah 1 → Klaster 1-4
      klaster     : parseInt(obj["label"]) + 1,
      u0          : parseFloat(obj["u_cluster_0"]),
      u1          : parseFloat(obj["u_cluster_1"]),
      u2          : parseFloat(obj["u_cluster_2"]),
      u3          : parseFloat(obj["u_cluster_3"]),
      region_type : obj["region_type"],       // "lower" atau "boundary"
      membership  : parseFloat(obj["membership_dominan"]),
      kepastian   : obj["kepastian"],
    };
  });
}

// =====================================================
// FUNGSI: Load SEMUA CSV sekaligus (2019–2025)
// =====================================================
async function loadSemuaData() {
  // Tampilkan tulisan loading sementara di tabel
  document.getElementById("isiTabel").innerHTML = `
    <tr>
      <td colspan="4" style="text-align:center;padding:30px;color:#aaa">
        ⏳ Memuat data, harap tunggu...
      </td>
    </tr>`;

  try {
    // Baca semua CSV secara paralel agar lebih cepat
    await Promise.all(TAHUN_LIST.map(async tahun => {
      semuaData[tahun] = await bacaCSV(tahun);
    }));

    // Setelah semua data siap, bangun dashboard
    buatTombolTahun();
    gantTahun(tahunAktif);

  } catch (err) {
    // Kalau ada error (mis. file CSV tidak ada / tidak pakai server)
    document.getElementById("isiTabel").innerHTML = `
      <tr>
        <td colspan="4" style="text-align:center;padding:30px;color:#c00">
          ❌ Gagal memuat data CSV.<br>
          Pastikan kamu membuka dashboard via <strong>Live Server</strong> atau <strong>XAMPP</strong>,
          bukan dengan dobel-klik file HTML.<br>
          <small style="color:#999">${err.message}</small>
        </td>
      </tr>`;
    console.error("Error memuat CSV:", err);
  }
}

// =====================================================
// FUNGSI: Buat tombol pilih tahun (2019-2025)
// =====================================================
function buatTombolTahun() {
  const kontainer = document.getElementById("tombolTahun");
  kontainer.innerHTML = "";

  TAHUN_LIST.forEach(tahun => {
    const btn = document.createElement("button");
    btn.className  = "btn-tahun" + (tahun === tahunAktif ? " aktif" : "");
    btn.textContent = tahun;
    btn.onclick    = () => gantTahun(tahun);
    kontainer.appendChild(btn);
  });
}

// =====================================================
// FUNGSI: Ganti tahun aktif → update semua konten
// =====================================================
function gantTahun(tahun) {
  tahunAktif = tahun;

  // Update tombol yang aktif
  document.querySelectorAll(".btn-tahun").forEach(btn => {
    btn.classList.toggle("aktif", parseInt(btn.textContent) === tahun);
  });

  // Update judul dan gambar peta
//   const judulPeta = document.getElementById("judulPeta");
  const gambarPeta = document.getElementById("gambarPeta");
  const pesanPeta  = document.getElementById("pesanPeta");

//   judulPeta.textContent = `Peta Klasterisasi FRCM per Provinsi - Tahun ${tahun}`;

  // Reset error state dulu sebelum ganti src
  gambarPeta.style.display = "block";
  pesanPeta.style.display  = "none";
  gambarPeta.src = `peta/Peta_Klaster_FRCM_${tahun}.png`;

  // Update judul tabel
  document.getElementById("judulTabel").textContent =
    `DAFTAR PROVINSI PER KLASTER (TAHUN ${tahun})`;

  // Isi ulang tabel
  isiTabel(tahun);
}

// =====================================================
// FUNGSI: Isi tabel dengan data tahun tertentu
// =====================================================
function isiTabel(tahun) {
  const tbody = document.getElementById("isiTabel");
  tbody.innerHTML = "";

  const data = semuaData[tahun];
  if (!data || data.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align:center;color:#aaa;padding:20px">
          Data untuk tahun ${tahun} tidak tersedia.
        </td>
      </tr>`;
    return;
  }

  // Urutkan: berdasarkan nomor klaster dulu, lalu nama A-Z
  const dataUrut = [...data].sort((a, b) =>
    a.klaster - b.klaster || a.provinsi.localeCompare(b.provinsi, "id")
  );

  dataUrut.forEach(item => {
    const tr = document.createElement("tr");

    const isBoundary = item.region_type === "boundary";

    // Bar membership: lebar proporsional dengan nilai membership
    const lebarBar = Math.round(item.membership * 100);

    const statusHTML = isBoundary
      ? `<span class="status-boundary">Boundary</span>`
      : `<span class="status-pasti">✓ Pasti</span>`;

    tr.innerHTML = `
      <td>${item.provinsi}</td>
      <td>
        <span class="badge-klaster badge-k${item.klaster}">K${item.klaster}</span>
      </td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="bar-membership">
            <div class="bar-isi" style="width:${lebarBar}%"></div>
          </div>
          <span style="font-size:20px;color:#555">${item.membership.toFixed(2)}</span>
        </div>
      </td>
      <td>${statusHTML}</td>
    `;

    // Klik baris → buka popup detail provinsi ini
    tr.onclick = () => bukaPopup(item.provinsi, tahun);

    tbody.appendChild(tr);
  });
}

// =====================================================
// FUNGSI: Buka popup detail provinsi
// =====================================================
function bukaPopup(namaProvinsi, tahun) {
  const data = semuaData[tahun];
  const item = data?.find(d => d.provinsi === namaProvinsi);
  if (!item) return;

  const warna    = WARNA_KLASTER[item.klaster];
  const isBoundary = item.region_type === "boundary";

  // Ambil data tren: klaster provinsi ini di tiap tahun
  // null jika tahun tidak ada datanya
  const dataTren = TAHUN_LIST.map(t => {
    const d = semuaData[t]?.find(d => d.provinsi === namaProvinsi);
    return d ? d.klaster : null;
  });

  // Data distribusi membership keempat klaster
  const distU = [
    { label: "K1", nilai: item.u0, warna: WARNA_KLASTER[1] },
    { label: "K2", nilai: item.u1, warna: WARNA_KLASTER[2] },
    { label: "K3", nilai: item.u2, warna: WARNA_KLASTER[3] },
    { label: "K4", nilai: item.u3, warna: WARNA_KLASTER[4] },
  ];

  // ---- Bangun HTML timeline dot tren ----
  // Tiap tahun ditampilkan sebagai kolom: label klaster → dot berwarna → label tahun
  const timelineHTML = TAHUN_LIST.map((t, i) => {
    const k      = dataTren[i];                          // nomor klaster tahun ini
    const w      = k ? WARNA_KLASTER[k] : "#ddd";       // warna dot
    const isAktif = t === tahun;                         // tahun yang sedang dilihat

    return `
      <div class="timeline-item">
        <!-- Label klaster di atas dot -->
        <div class="timeline-klabel" style="color:${w}">
          ${k ? `K${k}` : "–"}
        </div>
        <!-- Dot lingkaran berwarna -->
        <div class="timeline-dot ${isAktif ? "aktif" : ""}"
             style="background:${w}"
             title="${t}: ${k ? "Klaster " + k : "Data tidak ada"}">
          ${k ? k : "?"}
        </div>
        <!-- Label tahun di bawah dot -->
        <div class="timeline-tahun ${isAktif ? "aktif" : ""}">${t}</div>
      </div>`;
  }).join("");

  // ---- Isi popup ----
  document.getElementById("isiPopup").innerHTML = `

    <!-- Nama provinsi + tahun sebagai judul popup -->
    <div class="popup-nama" style="color:${warna}">
      ● ${namaProvinsi} — ${tahun}
    </div>

    <!-- 3 kartu: Klaster, Membership Dominan, Status Region -->
    <div class="popup-kartu-row">
      <div class="popup-kartu">
        KLASTER
        <div class="nilai k${item.klaster}">Klaster ${item.klaster}</div>
      </div>
      <div class="popup-kartu">
        MEMBERSHIP DOMINAN
        <div class="nilai">${item.membership.toFixed(4)}</div>
        <div style="background:#e0e0e0;height:6px;border-radius:3px;margin-top:7px">
          <div style="width:${item.membership*100}%;background:${warna};height:6px;border-radius:3px"></div>
        </div>
      </div>
      <div class="popup-kartu">
        STATUS REGION
        <div class="nilai" style="font-size:20px;margin-top:8px;color:${isBoundary ? "#888" : "#2e7d32"}">
          ${isBoundary ? "🔲 Boundary" : "✅ Pasti (Lower)"}
        </div>
      </div>
    </div>

    <!-- Distribusi nilai membership 4 klaster -->
    <div class="popup-seksi">DISTRIBUSI MEMBERSHIP</div>
    <div class="popup-kartu-row">
      ${distU.map(d => `
        <div class="popup-kartu">
          <div style="font-size:20px;color:#aaa">${d.label}</div>
          <div class="nilai" style="color:${d.warna}">${d.nilai.toFixed(3)}</div>
          <div style="background:#e0e0e0;height:5px;border-radius:3px;margin-top:6px">
            <div style="width:${d.nilai*100}%;background:${d.warna};height:5px;border-radius:3px"></div>
          </div>
        </div>
      `).join("")}
    </div>

    <!-- Timeline tren klaster 2019-2025 -->
    <!-- Dot berwarna = klaster, dot lebih besar = tahun yang sedang aktif -->
    <div class="popup-seksi">TREN KLASTER 2019–2025</div>
    <div class="timeline-tren">
      ${timelineHTML}
    </div>
  `;

  // Tampilkan overlay gelap dan popup
  document.getElementById("overlay").style.display = "block";
  document.getElementById("popup").style.display   = "block";
}

// =====================================================
// FUNGSI: Tutup popup
// =====================================================
function tutupPopup() {
  document.getElementById("overlay").style.display = "none";
  document.getElementById("popup").style.display   = "none";
}

// Tutup popup juga kalau tekan tombol Escape di keyboard
document.addEventListener("keydown", e => {
  if (e.key === "Escape") tutupPopup();
});

// =====================================================
// JALANKAN saat halaman pertama dibuka
// =====================================================
loadSemuaData();