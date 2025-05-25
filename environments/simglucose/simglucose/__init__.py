"""Package initialization for ``simglucose``.

This module optionally registers the Gym environment ``simglucose-v0``.  Some
development environments (including automated tests) may not have the ``gym``
package installed.  Additionally, importing this module multiple times should
not raise an error if the environment has already been registered.  The logic
below therefore guards the registration with ``try/except`` blocks and checks
whether the ID is already present in the registry.
"""

try:  # optional dependency
    from gym.envs.registration import register, registry

    already_registered = False
    try:
        # ``registry`` may not expose ``env_specs`` in all Gym versions
        if hasattr(registry, "env_specs"):
            already_registered = "simglucose-v0" in registry.env_specs
        else:
            registry.spec("simglucose-v0")
            already_registered = True
    except Exception:
        # ``spec`` raised an error -> not yet registered
        already_registered = False

    if not already_registered:
        register(id="simglucose-v0", entry_point="simglucose.envs:T1DSimEnv")
except Exception:  # gym is not installed or registration failed
    # In test environments without Gym we simply skip registration.
    pass
