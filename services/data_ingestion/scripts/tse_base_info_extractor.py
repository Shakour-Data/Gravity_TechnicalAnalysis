import json
import logging
import os
import ssl
from typing import Any

import finpy_tse as fpy
import pandas as pd

# TSETMC endpoints present non-standard certificates in some environments.
# Mirror the behaviour used in gravity_tse by disabling strict verification so
# the extractor can run in offline/air-gapped deployments.
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
try:  # pragma: no cover - platform dependent
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass


class TSEBaseInfoExtractor:
    def __init__(self, output_dir: str = "BasicTseInformation") -> None:
        self.output_dir = output_dir
        self.df: pd.DataFrame | None = None
        self.markets: pd.DataFrame | None = None
        self.panels: pd.DataFrame | None = None
        self.sectors: pd.DataFrame | None = None
        self.subsectors: pd.DataFrame | None = None
        self.companies: list[dict[str, Any]] | None = None

        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_stock_list(self) -> bool:
        """Load stock list from TSE"""
        self.logger.info("Fetching stock list from TSE...")
        try:
            self.df = fpy.Build_Market_StockList(
                bourse=True,
                farabourse=True,
                payeh=True,
                detailed_list=True,
                show_progress=True,
                save_excel=False,
                save_csv=False,
            )

            if self.df.empty:
                self.logger.error("No data received from TSE")
                return False

            self.logger.info(f"Successfully fetched {len(self.df)} stocks")
            self.df.reset_index(inplace=True)

            # Display available columns
            self.logger.debug(f"Available columns: {self.df.columns.tolist()}")
            return True

        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return False

    def get_market_mapping(self) -> dict[str, int]:
        """Get market ID mapping from existing JSON structure"""
        return {"بورس": 1, "فرابورس": 2}

    def get_panel_mapping(self) -> dict[str, int]:
        """Get panel ID mapping from existing JSON structure"""
        return {
            "بازار اول بورس": 1,
            "بازار دوم بورس": 2,
            "بازار اول (تابلوی اصلی) بورس": 1,
            "بازار اول (تابلوی فرعی) بورس": 2,
            "تابلوی اصلی فرابورس": 3,
            "تابلوی فرعی فرابورس": 4,
        }

    def get_sector_mapping(self) -> dict[str, int]:
        """Get sector ID mapping from existing JSON structure"""
        return {
            "هتل و رستوران": 55,
            "عرضه برق، گاز، بخارآب گرم": 40,
            "رایانه و فعالیت های وابسته به آن": 72,
            "بيمه و صندوق بازنشستگی به جزء تامین اجتماعی": 66,
            "فلزات اساسی": 27,
            "استخراج کانه های فلزی": 13,
            "مخابرات": 64,
            "سیمان، آهک و گچ": 53,
            "اطلاعات و ارتباطات": 73,
            "خرده فروشی،باستثنای وسایل نقلیه موتوری": 47,
            "فعالیتهای کمکی به نهادهای مالی واسط": 67,
            "ماشین آلات و دستگاه های برقی": 31,
            "مواد و محصولات دارویی": 43,
            "محصولات شیمیایی": 44,
            "لاستیک و پلاستیک": 25,
            "انبوه سازی، املاک و مستغلات": 70,
            "انتشار، چاپ و تکثیر": 22,
            "ساخت محصولات فلزی": 28,
            "محصولات چوبی": 20,
            "محصولات کاغذی": 21,
            "حمل ونقل، انبارداری و ارتباطات": 60,
            "استخراج نفت گاز و خدمات جنبی جز اکتشاف": 11,
            "خودرو و ساخت قطعات": 34,
            "زراعت و خدمات وابسته": 1,
            "سرمایه گذاریها": 56,
            "خدمات فنی و مهندسی": 74,
        }

    def get_subsector_mapping(self) -> dict[str, int]:
        """Get subsector ID mapping from existing JSON structure"""
        return {
            "هتل ها ، اردو و دیگر تدادرکات اقامت کوتاه": 5510,
            "تولید انتقال و توزیع نیروی برق": 4010,
            "مشاوره و تهیه نرم افزار": 7220,
            "بيمه غيرزندگي": 6603,
            "تولید فلزات گرانبهای غيرآهن": 2720,
            "استخراج آهن": 1310,
            "مخابرات": 6420,
            "تولید سیمان، آهک و گچ": 5394,
            "خدمات ارزش افزوده": 7310,
            "خرده فروشی انواع موادغذایی،نوشيدني وغيره": 4711,
            "بجز تامین وجوه بيمه و بازنشستگی": 6719,
            "تولید سیم و کابل عایق": 3130,
            "تولید ساير محصولات دارویی": 4399,
            "تولید مواد شيميايی پايه به جز کود": 4411,
            "تولید کد و ترکيبات نيتروژن": 4412,
            "تولید شوينده ها، عطر و محصولات آرايشی": 4424,
            "تولید تایر و بازسازی تایرهای لاستيکی": 2511,
            "تولید محصولات پلاستيکی": 2520,
            "تولید ساير محصولات لاستيکی": 2519,
            "املاک و مستغلات با ملک خود يا ليزينگ شده": 7010,
            "پيمانکاری املاک و مستغلات": 7021,
            "چاپ": 2221,
            "تولید ساير محصولات فلزی ساخته شده": 2899,
            "تولید تخته چند لا و ساير تخته ها": 2021,
            "تولید خمير کاغذ، کاغذ و مقوا": 2101,
            "تولید کاغذ و مقوا موجوددار و محفظه آنها": 2102,
            "حمل و نقل بار زمينی": 6023,
            "خدمات حمل و نقل دريايی": 6003,
            "خدمات جنبی استخراج نفت گاز جز اکتشاف": 1120,
            "ساير قطعات يدکی و جانبی وسايل نقليه موتوری": 3499,
            "قطعات يدکی و جانبی وسايل نقليه موتوری": 3430,
            "تولید وسايل نقليه موتوری": 3410,
            "کشاورزی،دامپروری و خدمات وابسته": 121,
            "پرورش حيوانات": 122,
            "پرورش طيور": 141,
            "ساير واسطه های مالی": 5699,
            "فعالیتهای ساختمانی و مشاوره فنی مرتبط": 7421,
            "تولید موتورها، مولدها و مبدلهای الکتريکی": 3110,
            "تولید تجهيزات توزيع و کنترل برق": 3120,
            "تولید لامپ برقی و تجهيزات روشنايی": 3150,
            "تولید،انتقال و توزیع برق،گاز،بخارآب گرم": 4011,
            "اداره ی بازارهای مالی": 6711,
            "فعالیتهای مرتبط با اوراق بهادار": 6712,
            "تولید داروهای شيميايی و گياهی": 4323,
            "تولید ماشین آلات کشاورزی و باغبانی": 2921,
            "تولید پمپ، کمپرسور، مته و دريچه": 2912,
        }

    def extract_markets(self) -> pd.DataFrame:
        """Extract market information"""
        self.logger.info("Extracting market information...")

        if "Market" not in self.df.columns:
            self.logger.error("Market column not found")
            return pd.DataFrame(columns=["MarketID", "MarketName"])

        markets = self.df[["Market"]].drop_duplicates().reset_index(drop=True)
        market_mapping = self.get_market_mapping()
        markets["MarketID"] = markets["Market"].map(market_mapping)

        markets = markets[["MarketID", "Market"]]
        markets.rename(columns={"Market": "MarketName"}, inplace=True)
        markets = markets.dropna(subset=["MarketID"])

        self.logger.info(f"Extracted {len(markets)} markets")
        return markets

    def extract_panels(self) -> pd.DataFrame:
        """Extract panel information"""
        self.logger.info("Extracting panel information...")

        if "Panel" not in self.df.columns:
            self.logger.error("Panel column not found")
            return pd.DataFrame(columns=["PanelID", "PanelName"])

        panels = self.df[["Panel"]].drop_duplicates().reset_index(drop=True)
        panel_mapping = self.get_panel_mapping()
        panels["PanelID"] = panels["Panel"].map(panel_mapping)

        panels = panels[["PanelID", "Panel"]]
        panels.rename(columns={"Panel": "PanelName"}, inplace=True)
        panels = panels.dropna(subset=["PanelID"])

        self.logger.info(f"Extracted {len(panels)} panels")
        return panels

    def extract_sectors(self) -> pd.DataFrame:
        """Extract sector information"""
        self.logger.info("Extracting sector information...")

        if "Sector" not in self.df.columns:
            self.logger.error("Sector column not found")
            return pd.DataFrame(columns=["SectorID", "SectorName"])

        sectors = self.df[["Sector"]].drop_duplicates().reset_index(drop=True)
        sector_mapping = self.get_sector_mapping()
        sectors["SectorID"] = sectors["Sector"].map(sector_mapping)

        sectors = sectors[["SectorID", "Sector"]]
        sectors.rename(columns={"Sector": "SectorName"}, inplace=True)
        sectors = sectors.dropna(subset=["SectorID"])

        self.logger.info(f"Extracted {len(sectors)} sectors")
        return sectors

    def extract_subsectors(self) -> pd.DataFrame:
        """Extract subsector information"""
        self.logger.info("Extracting subsector information...")

        if "Sub-Sector" not in self.df.columns:
            self.logger.warning("Sub-Sector column not found")
            # If no subsector, create from sectors
            if hasattr(self, "sectors") and self.sectors is not None:
                subsectors = self.sectors.copy()
                subsectors["SubSectorID"] = subsectors["SectorID"]
                subsectors.rename(columns={"SectorName": "SubSectorName"}, inplace=True)
                subsectors = subsectors[["SubSectorID", "SubSectorName", "SectorID"]]
                self.logger.info(f"Created {len(subsectors)} subsectors from sectors")
                return subsectors
            return pd.DataFrame(columns=["SubSectorID", "SubSectorName", "SectorID"])

        subsectors = self.df[["Sub-Sector", "Sector"]].drop_duplicates().reset_index(drop=True)
        subsector_mapping = self.get_subsector_mapping()
        subsectors["SubSectorID"] = subsectors["Sub-Sector"].map(subsector_mapping)

        # Map SectorID for foreign key relationship
        sector_map = self.sectors.set_index("SectorName")["SectorID"].to_dict()
        subsectors["SectorID"] = subsectors["Sector"].map(sector_map)

        subsectors = subsectors[["SubSectorID", "Sub-Sector", "SectorID"]]
        subsectors.rename(columns={"Sub-Sector": "SubSectorName"}, inplace=True)
        subsectors = subsectors.dropna(subset=["SubSectorID", "SectorID"])

        self.logger.info(f"Extracted {len(subsectors)} subsectors")
        return subsectors

    def extract_companies(self) -> list[dict[str, Any]]:
        """Extract company information using CompanyCode12 as ID"""
        self.logger.info("Extracting company information...")

        companies_data = []

        # Create mapping dictionaries
        market_map = self.markets.set_index("MarketName")["MarketID"].to_dict()
        panel_map = self.panels.set_index("PanelName")["PanelID"].to_dict()
        sector_map = self.sectors.set_index("SectorName")["SectorID"].to_dict()
        subsector_map = self.subsectors.set_index("SubSectorName")["SubSectorID"].to_dict()

        for idx, row in self.df.iterrows():
            # Use CompanyCode12 as CompanyID
            company_code12 = row.get("Company Code(12)", "")
            company_id = company_code12 if company_code12 else f"COMP_{idx + 1:04d}"

            company_record = {
                "CompanyID": company_id,
                "Ticker": row.get("Ticker", ""),
                "Name": row.get("Name", ""),
                "Name(EN)": row.get("Name(EN)", ""),
                "CompanyCode12": row.get("Company Code(12)", ""),
                "Ticker4": row.get("Ticker(4)", ""),
                "Ticker5": row.get("Ticker(5)", ""),
                "Ticker12": row.get("Ticker(12)", ""),
                "Comment": row.get("Comment", "-"),
                "SectorCode": row.get("Sector Code", ""),
                "SubSectorCode": row.get("Sub-Sector Code", ""),
                "PanelCode": row.get("Panel Code", ""),
            }

            # Add foreign keys
            if "Market" in self.df.columns and pd.notna(row.get("Market")):
                company_record["MarketID"] = market_map.get(row["Market"])

            if "Panel" in self.df.columns and pd.notna(row.get("Panel")):
                company_record["PanelID"] = panel_map.get(row["Panel"])

            if "Sector" in self.df.columns and pd.notna(row.get("Sector")):
                company_record["SectorID"] = sector_map.get(row["Sector"])

            if "Sub-Sector" in self.df.columns and pd.notna(row.get("Sub-Sector")):
                company_record["SubSectorID"] = subsector_map.get(row["Sub-Sector"])
            elif "Sector" in self.df.columns and pd.notna(row.get("Sector")):
                company_record["SubSectorID"] = sector_map.get(row["Sector"])

            companies_data.append(company_record)

        self.logger.info(f"Extracted {len(companies_data)} companies")
        return companies_data

    def save_to_json(self, data: Any, filename: str):
        """Save data to JSON file"""
        filepath = os.path.join(self.output_dir, filename)

        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient="records", indent=4, force_ascii=False)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Saved {filename}")

    def create_master_structure(self):
        """Create master JSON file showing complete structure"""
        self.logger.info("Creating master structure file...")

        master_data = {
            "metadata": {
                "total_companies": len(self.companies),
                "total_markets": len(self.markets),
                "total_panels": len(self.panels),
                "total_sectors": len(self.sectors),
                "total_subsectors": len(self.subsectors),
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "markets": self.markets.to_dict("records"),
            "panels": self.panels.to_dict("records"),
            "sectors": self.sectors.to_dict("records"),
            "subsectors": self.subsectors.to_dict("records"),
            "companies_sample": self.companies[:10],  # First 10 companies as sample
        }

        self.save_to_json(master_data, "master_structure.json")

    def validate_data(self):
        """Validate the extracted data"""
        self.logger.info("Validating extracted data...")

        try:
            # Check if all files are created and contain data
            required_files = [
                "markets.json",
                "panels.json",
                "sectors.json",
                "subsectors.json",
                "companies.json",
            ]

            for file in required_files:
                filepath = os.path.join(self.output_dir, file)
                if not os.path.exists(filepath):
                    self.logger.error(f"Missing file: {file}")
                    continue

                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)

                self.logger.info(f"{file}: {len(data)} records")

            # Display sample company data
            if self.companies:
                self.logger.info("Sample company data:")
                sample = self.companies[0]
                for key, value in list(sample.items())[:8]:
                    self.logger.info(f"   {key}: {value}")

            self.logger.info("Data validation completed successfully!")

        except Exception as e:
            self.logger.error(f"Validation error: {e}")

    def run_extraction(self):
        """Main method to run the complete extraction process"""
        self.logger.info("Starting TSE Data Extraction Process...")
        self.logger.info("=" * 50)

        # Step 1: Load stock list
        if not self.load_stock_list():
            return False

        # Step 2: Extract basic information
        self.markets = self.extract_markets()
        self.panels = self.extract_panels()
        self.sectors = self.extract_sectors()
        self.subsectors = self.extract_subsectors()

        # Step 3: Extract companies
        self.companies = self.extract_companies()

        # Step 4: Save all data to JSON files
        self.logger.info("Saving data to JSON files...")
        self.save_to_json(self.markets, "markets.json")
        self.save_to_json(self.panels, "panels.json")
        self.save_to_json(self.sectors, "sectors.json")
        self.save_to_json(self.subsectors, "subsectors.json")
        self.save_to_json(self.companies, "companies.json")

        # Step 5: Create master structure file
        self.create_master_structure()

        # Step 6: Validate data
        self.validate_data()

        self.logger.info("=" * 50)
        self.logger.info("TSE Data Extraction Completed Successfully!")
        self.logger.info(f"Output directory: {os.path.abspath(self.output_dir)}")

        return True


# Loading Functions
def load_json_file(filepath: str) -> list[dict[str, Any]] | None:
    """Load data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading JSON file {filepath}: {e}")
        return None


def load_tse_base_markets(data_dir: str = "BasicTseInformation") -> list[dict[str, Any]] | None:
    """Load TSE markets data from JSON file."""
    return load_json_file(os.path.join(data_dir, "markets.json"))


def load_tse_base_panels(data_dir: str = "BasicTseInformation") -> list[dict[str, Any]] | None:
    """Load TSE panels data from JSON file."""
    return load_json_file(os.path.join(data_dir, "panels.json"))


def load_tse_base_sectors(data_dir: str = "BasicTseInformation") -> list[dict[str, Any]] | None:
    """Load TSE sectors data from JSON file."""
    return load_json_file(os.path.join(data_dir, "sectors.json"))


def load_tse_base_subsectors(data_dir: str = "BasicTseInformation") -> list[dict[str, Any]] | None:
    """Load TSE subsectors data from JSON file."""
    return load_json_file(os.path.join(data_dir, "subsectors.json"))


def load_tse_base_companies(data_dir: str = "BasicTseInformation") -> list[dict[str, Any]] | None:
    """Load TSE companies data from JSON file."""
    return load_json_file(os.path.join(data_dir, "companies.json"))


def load_tse_base_master_structure(data_dir: str = "BasicTseInformation") -> dict[str, Any] | None:
    """Load TSE master structure data from JSON file."""
    filepath = os.path.join(data_dir, "master_structure.json")
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading JSON file {filepath}: {e}")
        return None


if __name__ == "__main__":
    extractor = TSEBaseInfoExtractor()
    success = extractor.run_extraction()
    if not success:
        print("Extraction failed.")
