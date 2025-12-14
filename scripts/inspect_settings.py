from gravity_tech.config import settings as settings_module

print('Has Settings class:', hasattr(settings_module, 'Settings'))
Settings = settings_module.Settings
print('class attr present:', 'expose_db_explorer' in Settings.__dict__)
print('class dir contains expose_db_explorer:', [n for n in dir(Settings) if 'expose_db_explorer' in n])
print('instance attr present:', hasattr(settings_module.settings, 'expose_db_explorer'))
print('instance attr value:', getattr(settings_module.settings, 'expose_db_explorer', None))
print('repr Settings:', repr(Settings))
print('repr settings instance:', settings_module.settings)
