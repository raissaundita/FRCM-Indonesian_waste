// =====================================================
// MAIN DASHBOARD SCRIPT
// Reads CSV data by year, fills the table, and displays
// province details with a cluster trend timeline
// =====================================================

// Currently active year
let tahunAktif = 2022;

// Store all loaded data
// Example: semuaData[2019] = [{ provinsi, klaster, ... }, ...]
const semuaData = {};

// Cluster colors
const WARNA_KLASTER = {
  1: "#29B6F6",  // blue
  2: "#66BB6A",  // green
  3: "#FBC02D",  // yellow
  4: "#EF5350",  // red
};

const TAHUN_LIST = [2019, 2020, 2021, 2022, 2023, 2024, 2025];


// =====================================================
// FUNCTION: Read one CSV file
// =====================================================
async function bacaCSV(tahun) {

  const response = await fetch(`data/klaster_${tahun}.csv`);
  const teks = await response.text();

  const baris = teks.trim().split("\n");

  // Remove BOM if present
  const header = baris[0]
    .replace(/^\uFEFF/, "")
    .split(",");

  return baris.slice(1).map(b => {

    const kolom = b.split(",");
    const obj = {};

    header.forEach((h, i) => {
      obj[h.trim()] = kolom[i]?.trim();
    });

    return {
      provinsi: obj["Provinsi"],

      // CSV labels are 0–3, converted to Cluster 1–4
      klaster: parseInt(obj["label"]) + 1,

      u0: parseFloat(obj["u_cluster_0"]),
      u1: parseFloat(obj["u_cluster_1"]),
      u2: parseFloat(obj["u_cluster_2"]),
      u3: parseFloat(obj["u_cluster_3"]),

      region_type: obj["region_type"],

      membership: parseFloat(obj["membership_dominan"]),

      kepastian: obj["kepastian"],
    };
  });
}


// =====================================================
// FUNCTION: Load all CSV files (2019–2025)
// =====================================================
async function loadSemuaData() {

  document.getElementById("isiTabel").innerHTML = `
    <tr>
      <td
        colspan="4"
        style="text-align:center;padding:30px;color:#aaa"
      >
        ⏳ Loading data, please wait...
      </td>
    </tr>
  `;

  try {

    await Promise.all(
      TAHUN_LIST.map(async tahun => {
        semuaData[tahun] = await bacaCSV(tahun);
      })
    );

    buatTombolTahun();

    gantTahun(tahunAktif);

  } catch (err) {

    document.getElementById("isiTabel").innerHTML = `
      <tr>
        <td
          colspan="4"
          style="text-align:center;padding:30px;color:#c00"
        >
          ❌ Failed to load CSV data.<br>
          Make sure the data files are available in the
          <strong>data</strong> folder.<br>
          <small style="color:#999">
            ${err.message}
          </small>
        </td>
      </tr>
    `;

    console.error("Error loading CSV:", err);
  }
}


// =====================================================
// FUNCTION: Create year buttons
// =====================================================
function buatTombolTahun() {

  const kontainer = document.getElementById("tombolTahun");

  kontainer.innerHTML = "";

  TAHUN_LIST.forEach(tahun => {

    const btn = document.createElement("button");

    btn.className =
      "btn-tahun" +
      (tahun === tahunAktif ? " aktif" : "");

    btn.textContent = tahun;

    btn.onclick = () => gantTahun(tahun);

    kontainer.appendChild(btn);
  });
}


// =====================================================
// FUNCTION: Change active year
// =====================================================
function gantTahun(tahun) {

  tahunAktif = tahun;


  // Update active year button
  document
    .querySelectorAll(".btn-tahun")
    .forEach(btn => {

      btn.classList.toggle(
        "aktif",
        parseInt(btn.textContent) === tahun
      );

    });


  // ===================================================
  // UPDATE MAP
  // ===================================================

  const gambarPeta =
    document.getElementById("gambarPeta");

  const pesanPeta =
    document.getElementById("pesanPeta");


  // Reset map display
  gambarPeta.style.display = "block";

  pesanPeta.style.display = "none";


  /*
    PRIMARY FILE NAME:

    maps/FRCM cluster maps 2022.png

    FALLBACK FILE NAME:

    maps/FRCM clustering maps 2022.png

    This fallback is included so the dashboard still works
    if the uploaded files use the older "clustering" name.
  */

  const mapUtama =
    `maps/FRCM cluster maps ${tahun}.png`;

  const mapFallback =
    `maps/FRCM clustering maps ${tahun}.png`;


  // Reset fallback status
  delete gambarPeta.dataset.fallback;


  gambarPeta.onerror = () => {

    // Try alternative filename
    if (gambarPeta.dataset.fallback !== "used") {

      gambarPeta.dataset.fallback = "used";

      gambarPeta.src = mapFallback;

      return;
    }


    // If both filenames fail
    gambarPeta.style.display = "none";

    pesanPeta.innerHTML = `
      ⚠️ The map image is not available.<br>
      Make sure the map file exists in the
      <strong>maps</strong> folder.<br><br>

      Expected file name:<br>

      <strong>
        FRCM cluster maps ${tahun}.png
      </strong>
      <br><br>

      File names are case-sensitive on the hosting server.
    `;

    pesanPeta.style.display = "flex";
  };


  // Load map
  gambarPeta.src = mapUtama;


  // ===================================================
  // UPDATE TABLE TITLE
  // ===================================================

  document
    .getElementById("judulTabel")
    .textContent =
      `LIST OF PROVINCES BY CLUSTER (YEAR ${tahun})`;


  // Update table contents
  isiTabel(tahun);
}


// =====================================================
// FUNCTION: Fill province table
// =====================================================
function isiTabel(tahun) {

  const tbody =
    document.getElementById("isiTabel");

  tbody.innerHTML = "";


  const data = semuaData[tahun];


  if (!data || data.length === 0) {

    tbody.innerHTML = `
      <tr>
        <td
          colspan="4"
          style="
            text-align:center;
            color:#aaa;
            padding:20px
          "
        >
          Data for ${tahun} is not available.
        </td>
      </tr>
    `;

    return;
  }


  // Sort by cluster number, then province name
  const dataUrut = [...data].sort(
    (a, b) =>
      a.klaster - b.klaster ||
      a.provinsi.localeCompare(
        b.provinsi,
        "en"
      )
  );


  dataUrut.forEach(item => {

    const tr =
      document.createElement("tr");


    const isBoundary =
      item.region_type === "boundary";


    // Membership bar width
    const lebarBar =
      Math.round(item.membership * 100);


    const statusHTML = isBoundary

      ? `
        <span class="status-boundary">
          Boundary
        </span>
      `

      : `
        <span class="status-pasti">
          ✓ Certain
        </span>
      `;


    tr.innerHTML = `

      <td>
        ${item.provinsi}
      </td>


      <td>

        <span
          class="
            badge-klaster
            badge-k${item.klaster}
          "
        >
          K${item.klaster}
        </span>

      </td>


      <td>

        <div
          style="
            display:flex;
            align-items:center;
            gap:8px
          "
        >

          <div class="bar-membership">

            <div
              class="bar-isi"
              style="
                width:${lebarBar}%
              "
            ></div>

          </div>

          <span
            style="
              font-size:20px;
              color:#555
            "
          >
            ${item.membership.toFixed(2)}
          </span>

        </div>

      </td>


      <td>
        ${statusHTML}
      </td>

    `;


    // Click row to open province details
    tr.onclick = () =>
      bukaPopup(
        item.provinsi,
        tahun
      );


    tbody.appendChild(tr);
  });
}


// =====================================================
// FUNCTION: Open province detail popup
// =====================================================
function bukaPopup(
  namaProvinsi,
  tahun
) {

  const data =
    semuaData[tahun];


  const item =
    data?.find(
      d => d.provinsi === namaProvinsi
    );


  if (!item) return;


  const warna =
    WARNA_KLASTER[item.klaster];


  const isBoundary =
    item.region_type === "boundary";


  // ===================================================
  // CLUSTER TREND 2019–2025
  // ===================================================

  const dataTren =
    TAHUN_LIST.map(t => {

      const d =
        semuaData[t]?.find(
          d => d.provinsi === namaProvinsi
        );

      return d
        ? d.klaster
        : null;
    });


  // ===================================================
  // MEMBERSHIP DISTRIBUTION
  // ===================================================

  const distU = [

    {
      label: "K1",
      nilai: item.u0,
      warna: WARNA_KLASTER[1]
    },

    {
      label: "K2",
      nilai: item.u1,
      warna: WARNA_KLASTER[2]
    },

    {
      label: "K3",
      nilai: item.u2,
      warna: WARNA_KLASTER[3]
    },

    {
      label: "K4",
      nilai: item.u3,
      warna: WARNA_KLASTER[4]
    },

  ];


  // ===================================================
  // BUILD CLUSTER TREND TIMELINE
  // ===================================================

  const timelineHTML =
    TAHUN_LIST
      .map((t, i) => {

        const k =
          dataTren[i];


        const w =
          k
            ? WARNA_KLASTER[k]
            : "#ddd";


        const isAktif =
          t === tahun;


        return `

          <div class="timeline-item">

            <div
              class="timeline-klabel"
              style="color:${w}"
            >
              ${k ? `K${k}` : "–"}
            </div>


            <div
              class="
                timeline-dot
                ${isAktif ? "aktif" : ""}
              "
              style="
                background:${w}
              "
              title="
                ${t}: ${
                  k
                    ? "Cluster " + k
                    : "No data"
                }
              "
            >

              ${k ? k : "?"}

            </div>


            <div
              class="
                timeline-tahun
                ${isAktif ? "aktif" : ""}
              "
            >
              ${t}
            </div>

          </div>

        `;

      })
      .join("");


  // ===================================================
  // POPUP CONTENT
  // ===================================================

  document
    .getElementById("isiPopup")
    .innerHTML = `


      <!-- Province name and selected year -->

      <div
        class="popup-nama"
        style="color:${warna}"
      >

        ● ${namaProvinsi} — ${tahun}

      </div>


      <!-- Main information cards -->

      <div class="popup-kartu-row">


        <div class="popup-kartu">

          CLUSTER

          <div
            class="
              nilai
              k${item.klaster}
            "
          >
            Cluster ${item.klaster}
          </div>

        </div>


        <div class="popup-kartu">

          DOMINANT MEMBERSHIP

          <div class="nilai">

            ${item.membership.toFixed(4)}

          </div>


          <div
            style="
              background:#e0e0e0;
              height:6px;
              border-radius:3px;
              margin-top:7px
            "
          >

            <div
              style="
                width:${item.membership * 100}%;
                background:${warna};
                height:6px;
                border-radius:3px
              "
            ></div>

          </div>

        </div>


        <div class="popup-kartu">

          REGION STATUS

          <div
            class="nilai"
            style="
              font-size:20px;
              margin-top:8px;
              color:${
                isBoundary
                  ? "#888"
                  : "#2e7d32"
              }
            "
          >

            ${
              isBoundary
                ? "🔲 Boundary"
                : "✅ Certain (Lower)"
            }

          </div>

        </div>


      </div>


      <!-- Membership distribution -->

      <div class="popup-seksi">

        MEMBERSHIP DISTRIBUTION

      </div>


      <div class="popup-kartu-row">

        ${distU.map(d => `

          <div class="popup-kartu">

            <div
              style="
                font-size:20px;
                color:#aaa
              "
            >
              ${d.label}
            </div>


            <div
              class="nilai"
              style="
                color:${d.warna}
              "
            >
              ${d.nilai.toFixed(3)}
            </div>


            <div
              style="
                background:#e0e0e0;
                height:5px;
                border-radius:3px;
                margin-top:6px
              "
            >

              <div
                style="
                  width:${d.nilai * 100}%;
                  background:${d.warna};
                  height:5px;
                  border-radius:3px
                "
              ></div>

            </div>

          </div>

        `).join("")}

      </div>


      <!-- Cluster trend -->

      <div class="popup-seksi">

        CLUSTER TREND 2019–2025

      </div>


      <div class="timeline-tren">

        ${timelineHTML}

      </div>

    `;


  // Show popup and dark overlay
  document
    .getElementById("overlay")
    .style.display = "block";


  document
    .getElementById("popup")
    .style.display = "block";
}


// =====================================================
// FUNCTION: Close popup
// =====================================================
function tutupPopup() {

  document
    .getElementById("overlay")
    .style.display = "none";


  document
    .getElementById("popup")
    .style.display = "none";
}


// Close popup using Escape key
document.addEventListener(
  "keydown",
  e => {

    if (e.key === "Escape") {
      tutupPopup();
    }

  }
);


// =====================================================
// RUN DASHBOARD
// =====================================================

loadSemuaData();
