from pneuma_seeker.services.skill_based_core.skills.base import SkillBase, SkillResult


class RespondSkill(SkillBase):
    name = "respond"

    def execute(self, args: dict, agent) -> SkillResult:
        message = args.get("message", "")
        if not message:
            return SkillResult("Error: 'message' is required.")
        agent._user_response = message
        return SkillResult(f"Response sent to user: {message}", terminates=True)
