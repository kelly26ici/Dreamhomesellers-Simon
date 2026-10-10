from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropertyCard:
    id: str
    title: str
    description: str
    property_type: str
    listing_type: str
    price_period: str
    status: str
    price: float
    currency: str
    bedrooms: int | None
    bathrooms: int | None
    square_meters: float | None
    lot_size_sqm: float | None
    location: str
    address: str | None
    town: str | None
    city: str
    county: str | None
    country: str
    amenities: list[str] = field(default_factory=list)
    furnished: bool = False
    year_built: int | None = None
    video_url: str | None = None
    agent_id: str | None = None
    external_id: str | None = None

    @property
    def price_per_sqm(self) -> float | None:
        if self.square_meters and self.square_meters > 0:
            return round(self.price / self.square_meters, 2)
        return None


@dataclass(frozen=True)
class AgentProfile:
    id: str
    first_name: str
    last_name: str | None
    email: str | None
    phone: str | None
    agency_name: str | None
    is_verified: bool


@dataclass(frozen=True)
class PropertyImage:
    id: str
    property_id: str
    url: str
    sort_order: int
    is_featured: bool


@dataclass(frozen=True)
class ViewingSlot:
    id: str
    property_id: str | None
    customer_phone: str
    customer_name: str | None
    viewing_date: str
    duration_minutes: int
    status: str
    notes: str | None


@dataclass(frozen=True)
class Lead:
    id: str
    customer_phone: str
    customer_name: str | None
    customer_email: str | None
    property_id: str | None
    agent_id: str | None
    inquiry_type: str
    message: str | None
