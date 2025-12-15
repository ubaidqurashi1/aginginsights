# Run validation
validator = AgingValidation()
results = validator.run_full_validation(data)

# Display results
import json
print(json.dumps(results, indent=2, default=str))