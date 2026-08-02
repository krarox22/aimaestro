"""What aiMaestro remembers, and the tool it uses to decide to remember."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """What aiMaestro knows about the person it is talking to."""

    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="Where the user lives")
    job: Optional[str] = Field(default=None, description="What the user does for work")
    connections: list[str] = Field(
        default_factory=list,
        description="People in the user's life: family, friends, coworkers",
    )
    interests: list[str] = Field(
        default_factory=list, description="Things the user is interested in"
    )


class ToDo(BaseModel):
    """A single task on the user's list."""

    task: str = Field(description="The task to be completed")
    time_to_complete: Optional[int] = Field(
        default=None, description="Estimated minutes to complete"
    )
    deadline: Optional[datetime] = Field(
        default=None, description="When this needs to be done, if there is a date"
    )
    solutions: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, actionable next steps: specific services, contacts, or "
            "options relevant to finishing the task"
        ),
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        default="not started", description="Current status of the task"
    )


class UpdateMemory(TypedDict):
    """Signal that something in this conversation is worth remembering."""

    update_type: Literal["user", "todo", "instructions"]
